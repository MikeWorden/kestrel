"""
KestrelEngine  (Retrieval-Augmented Generation) implementation using OpenAI and ChromaDB.

Pipeline:
    1. load()               - Read documents from disk
    2. chunk()              - Split documents into overlapping chunks
    3. embed()              - Generate embeddings via OpenAI
    4. index()              - Store chunks + embeddings in ChromaDB
    5. retrieve()           - Query ChromaDB for top-k relevant chunks
    6. generate_prompt()    - Build prompt with retrieved chunks + user query
    7. query()              - Call LLM with prompt and return answer + sources
    8. validate_connection() - End-to-end test of the full RAG pipeline

import logging
from pathlib import Path
from typing import Optional
from loader import Loader
import re
import tiktoken
from tqdm import tqdm


import chromadb
import json
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import gradio as gr

from config import Config

"""


class KestrelEngine:
    """
    Retrieval-Augmented Generation over NVD using a ChromaDB vector store and OpenAI LLM.
    Usage:
        config = Config()
        rag = RAG(config)
        rag.build_index()               # one-time index build
        answer = rag.query("your question here")
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client: OpenAI = config.client          # OpenAI/OpenRouter client
        self.chroma_client = None                    # set in _init_chroma()
        self.collection = None                       # set in _init_chroma()
        self._init_chroma()
        self.docs = []                                  # raw loaded documents
        self.chunks = []                                # chunked documents ready for embedding 
        self.loader = Loader(config)                          # data loader instance
        self.header_pattern = re.compile(
            r'^(#{1,4}\s.+|[A-Z][A-Za-z\s]{10,}:|[A-Z\s]{10,})$',
            re.MULTILINE
        )
        self.enc = tiktoken.get_encoding("cl100k_base")  # matches text-embedding-3-large
        self.history  =  []

   
    # Initialization
   

    def _init_chroma(self) -> None:
        """Connect to (or create) the ChromaDB persistent store."""
        
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_dir)
        # self.collection = self.chroma_client.get_or_create_collection(name=self.config.collection_name)

        
        openai_ef = OpenAIEmbeddingFunction(
            api_key=self.config.openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
            model_name=self.config.embed_model, # Specify the exact OpenRouter model path
            dimensions=512
        )
        
        try:
            self.collection = self.chroma_client.get_collection(name="my_document_collection",
                                                           embedding_function=openai_ef)
        except:
            self.collection = self.chroma_client.create_collection(name="my_document_collection",
                                                              embedding_function=openai_ef)
        
    # Private data loading and processing methods
   

    
   
    # Public interface
   
    def load(self) -> None:
        """
        Read documents from config.docs_dir.
        Returns a list of dicts: [{"source": filename, "text": content}, ...]
        """
        # Load KEV data        
        self.loader.load_kev_data()
        # Load NVD data
        self.loader.load_nvd_data()
        # Merge KEV and NVD data into enriched documents
        self.docs = self.loader.merge_kev_nvd()
        # Load PDF data
        self.loader.load_pdf_data()
        # Add PDF page documents to self.docs
        self.docs.extend(self.loader.pdf_docs)
        

    def chunk(self) -> None:
        self.chunk_json()  # chunk KEV and NVD documents
        self.chunk_pdf(self.loader.pdf_docs)  # chunk PDF page documents


    def chunk_json(self) -> None:
        """
        Split documents into overlapping text chunks.
        Uses config.chunk_size and config.chunk_overlap.
        Returns a list of dicts: [{"source": ..., "chunk_id": ..., "text": ...}, ...]
        """
        
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        chunks = []
                
        for doc in self.docs:   
            text = doc["text"]
            source = doc["id"]
            metadata = doc["metadata"]
            
            record_source = doc.get("record_source", "")
            words = text.split()
  
            
            # Generally speaking, CVE descriptions are short enough to fit within our chunk size.
            if len(words) <= chunk_size:
                chunks.append({
                    "source": source,
                    "chunk_id": f"{source}_0",
                    "text": text,
                    "metadata": metadata
                })
                
                continue
            
            # But for longer texts, we apply chunking with overlap to preserve context across chunks.
            start = 0
            chunk_num = 0
            end = chunk_size
            while start < len(words):
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "source": source,
                    "chunk_id": f"{source}_{chunk_num}",
                    "text": chunk_text,
                    "metadata": metadata
                })
                start += chunk_size - chunk_overlap
                end = min(start + chunk_size, len(words))
                
                if end == len(words):
                    break  # avoid creating an empty chunk at the end
                
                start = end - chunk_overlap  # ensure overlap for next chunk
                chunk_num += 1

        self.chunks = chunks
        


    def chunk_pdf(self, docs: list[dict]) -> list[dict]:
        """
        Chunk PDF page docs by headers first, then subdivide by token count.
 
        Pipeline per page:
            1. Detect headers → split into (header, body) sections
            2. If section <= max_tokens → single chunk
            3. If section > max_tokens  → overlapping token windows
            4. Section header prepended to every sub-chunk for context
 
        Args:
            docs: Output of load() — list of page-level dicts.
 
        Returns:
            list[dict]: Chunked dicts with section and chunk metadata appended.
        """
        chunks = []
 
        for doc in docs:
            sections = self._split_on_headers(doc["text"])
 
            for header, body in sections:
                prefix    = f"{header}\n" if header else ""
                full_text = prefix + body
                tokens    = self.enc.encode(full_text)
 
                if len(tokens) <= self.config.chunk_size:
                    # Section fits within token budget — single chunk
                    chunks.append(self._make_chunk(doc, full_text, header, 0))
                else:
                    # Section too large — subdivide with overlap
                    sub_texts = self._subdivide(tokens, prefix)
                    for i, sub_text in enumerate(sub_texts):
                        chunks.append(self._make_chunk(doc, sub_text, header, i))
 
        print(f"PDF chunking: {len(docs)} pages → {len(chunks)} chunks")
        return chunks
 

   
    # Private: chunking
   
 
    def _split_on_headers(self, text: str) -> list[tuple[str, str]]:
        """
        Split text into (header, body) tuples on detected header lines.
        Covers markdown headers, Title Case labels, and ALL CAPS headings.
 
        Args:
            text: Cleaned page text.
 
        Returns:
            list[tuple[str, str]]: (header, body) pairs. Header may be empty
                                   for content before the first detected header.
        """
        sections = []
        matches  = list(self.header_pattern.finditer(text))
 
        if not matches:
            return [("", text)]     # no headers detected — single section
 
        # Content before first header
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble))
 
        for i, match in enumerate(matches):
            header = match.group().strip()
            start  = match.end()
            end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body   = text[start:end].strip()
            if body:
                sections.append((header, body))
 
        return sections
 
    def _subdivide(self, tokens: list, prefix: str) -> list[str]:
        """
        Subdivide a token list into overlapping windows.
        The section header prefix is prepended to every sub-chunk
        so each chunk retains its structural context.
 
        Args:
            tokens: Full token list for the section (prefix + body).
            prefix: Section header string prepended to each sub-chunk.
 
        Returns:
            list[str]: Decoded text strings, one per sub-chunk.
        """
        prefix_tokens = self.enc.encode(prefix)
        body_tokens   = tokens[len(prefix_tokens):]
        window        = self.config.chunk_size - len(prefix_tokens)
        sub_chunks    = []
        start         = 0
 
        while start < len(body_tokens):
            end       = min(start + window, len(body_tokens))
            chunk_tok = prefix_tokens + body_tokens[start:end]
            sub_chunks.append(self.enc.decode(chunk_tok))
 
            if end == len(body_tokens):
                break
            start += window - self.config.chunk_overlap
 
        return sub_chunks
 
    def _make_chunk(
        self,
        doc:       dict,
        text:      str,
        header:    str,
        chunk_num: int,
    ) -> dict:
        """
        Assemble a chunk dict matching the shape expected by index().
 
        Args:
            doc:       Source page dict from load()
            text:      Chunk text (header prefix + body window)
            header:    Section header string for metadata
            chunk_num: Sub-chunk index within this section
 
        Returns:
            dict: {"id", "text", "metadata"} ready for ChromaDB upsert.
        """
        # Sanitize header for use in ID (no spaces or special chars)
        safe_header = re.sub(r"\W+", "_", header[:30]).strip("_")
        chunk_id    = f"{doc['id']}__{safe_header}__c{chunk_num}" if safe_header \
                      else f"{doc['id']}__c{chunk_num}"
 
        return {
            "id":   chunk_id,
            "text": text,
            "doc_type": "pdf",
            "metadata": {
                **doc["metadata"],
                "section": header,
                "chunk":   chunk_num,
            }
        }


    def list_chunks(self, chunks: list[dict], max_lines: int = 10) -> str:
        """
        Utility to format a list of chunk dicts for display.
        Shows source, score, and text preview (first 100 chars).
        Truncates long lists with an ellipsis.

        Args:
            chunks: List of chunk dicts from retrieve() or query()["chunks"].
            max_lines: Max number of lines to display before truncating.

        Returns:
            str: Formatted string with one line per chunk.
        """
        lines = []
        for c in chunks:
            source = c.get("source", "unknown")
            score  = c.get("score", 0)
            meta   = c.get("metadata", {})
            print(f"Debug: {meta}")
            
    def _sanitize_metadata(self, meta: dict) -> dict:
        """Flatten any non-scalar values for ChromaDB compatibility."""
        clean = {}
        for k, v in meta.items():
            if isinstance(v, dict):
                continue                        # drop nested dicts entirely
            elif isinstance(v, list):
                clean[k] = ", ".join(str(i) for i in v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    
    def index(self) -> None:
        """
        Embed chunks and upsert into ChromaDB collection.
        Stores chunk text and source metadata alongside each vector.
        """
        chunks = self.chunks  # get chunked documents ready for embedding and indexing
        ids = [f"{c['source']}__{c['chunk_id']}" for c in chunks] # list of dicts: {"source": "privacy_policy.txt", "chunk_idx": 0}
        
        # Check for existing chunk IDs in the collection to avoid duplicates
        existing = self.collection.get(ids=ids, include=[]) # include -> means dont include optional fields like docs and metadata. Returns something like: {"ids": ["file1__0", "file2__3"]}
        existing_ids = set(existing["ids"]) # extract just ids,  create set from list, allows faster lookup, can check membership. e.g., chunk_id in existing_ids
        new_chunks = [
            (chunk_id, chunk)
            for chunk_id, chunk in zip(ids, chunks)
            if chunk_id not in existing_ids
        ]

        if not new_chunks:
            print("[index] No new chunks to index.")
            return

        # Added this to provide progress feedback during indexing, especially for large datasets.
        batch_size = 100

        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Indexing chunks"):
            batch = new_chunks[i:i + batch_size]
            self.collection.add(
                ids=       [chunk_id for chunk_id, _ in batch],
                documents= [chunk["text"] for _, chunk in batch],
                metadatas=[self._sanitize_metadata(chunk.get("metadata", {})) or {"document_source": "unknown"} for _, chunk in batch
]           )
        
        

   
    # Query pipeline
   

    
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Embed the query and retrieve top-k chunks from ChromaDB.
        Uses config.top_k.
        Returns a list of dicts: [{"text", "source", "doc_type", "score"}, ...]
        """
        if top_k is None:
            top_k = self.config.top_k

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert L2 distance to cosine similarity (0-1, higher = better match)
            score = round(1 / (1 + dist), 3)

            chunks.append({
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "doc_type": meta.get("doc_type", "nvd"),   # "kev", "nvd", or "pdf"
                "score":    score,
                "distance": round(dist, 4),                 # raw distance for debugging
                "metadata": meta,                           # full metadata for visualize.py
            })

        return chunks

    def generate_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """
        Assemble the prompt sent to the Responses API.
    
        Structure:
            [Retrieved context chunks]
            [User question]
    
        The system instruction is passed separately via `instructions` in
        generate_answer(), keeping system and user content clearly separated.
        """
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        ) # newlines create separation between chunks, ---
        return (
            f"### Retrieved Context:\n{context}\n\n"
            f"### Question:\n{query}"
        )

    def query(self, query: str, context_chunks: list[dict]) -> dict:
        """
        Full RAG pipeline: retrieve relevant chunks then generate answer with context.
        Retrieve → build grounded prompt → call LLM → return structured result.
        """
        client        = self.config.client
        llm_model     = self.config.llm_model
        system_prompt = self.config.system_prompt
        

        
        
        context = "\n\n".join(c["text"] for c in context_chunks)
        sources = sorted({c["source"] for c in context_chunks})

        # Step 1 — build grounded input (context + question)
        grounded_input = (
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        
        # Step 3 — build message list with history sandwiched in
        
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.history[-6:]                                   # last 3 turns (user + assistant)
            + [{"role": "user", "content": grounded_input}]
        )
        
        # Step  — call LLM via standard chat completions (OpenRouter compatible)
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )

        return {
            "question": query,
            "answer":   response.choices[0].message.content.strip(),
            "sources":  sources,
            "chunks":   context_chunks,
        }

   
    # Utilities
   

    def collection_size(self) -> int:
        """Return the number of chunks currently indexed."""
        return self.collection.count() if self.collection else 0
        

    def clear_index(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        if self.collection:
            self.collection.delete()

    def __repr__(self) -> str:
        size = self.collection_size() if self.collection else "uninitialized"
        return (
            f"<RAG collection={self.config.collection_name!r} "
            f"docs={self.config.docs_dir!r} chunks_indexed={size}>"
        )
        
        
    def validate_connection(self) -> bool:
        """
        Validate the full RAG pipeline end-to-end:
        1. ChromaDB collection is reachable and has documents
        2. Retrieval returns chunks
        3. LLM generates a grounded answer
        """
        try:
            # Step 1 — ChromaDB
            count = self.collection_size()
            if count == 0:
                print("Validation failed: collection is empty — run build_index() first")
                return False
            print(f"  [1/3] ChromaDB OK — {count} documents indexed")

            # Step 2 — Retrieval
            test_query = "Show me chunks where document_source = 'unknown'."  
            chunks     = self.retrieve(test_query, top_k=3)
            if not chunks:
                print("Validation failed: retrieve() returned no chunks")
                return False
            print(f"  [2/3] Retrieval OK — {len(chunks)} chunks returned")

            # Step 3 — LLM generation
            result = self.query(test_query, chunks)
            answer = result.get("answer", "").strip()
            if not answer:
                print("Validation failed: query() returned empty answer")
                return False
            print(f"  [3/3] LLM OK — answer preview: {answer[:120]}...")

            print("Validation passed — KestrelEngine is operational")
            return True

        except Exception as e:
            print(f"Validation failed: {e}")
            return False





        
def main():
    config = Config()
    if config.client is None:
        print("Error: OpenAI client not initialized. Please set OPENROUTER_API_KEY in your environment.")
        return
    else:
        # Intialization 
        print("OpenAI client initialized successfully!")
        kestrel = KestrelEngine(config)
        print("KestrelEngine initialized successfully!")
        # Load data and chunk 
        kestrel.load()
        print(f"Loaded {len(kestrel.docs)} documents from disk.")
        # print("Loading data and building index...")
        print(f"Chunking documents into pieces of ~{config.chunk_size} words with {config.chunk_overlap} words overlap...")
        kestrel.chunk()
        print(f"Chunking complete! Generated {len(kestrel.chunks)} chunks ready for embedding and indexing.")
        # Build Index
        kestrel.index()
        print("Indexing complete!")
        # Validate end-to-end pipeline with a test query        
        kestrel.validate_connection()
        
if __name__ == "__main__":
    main()