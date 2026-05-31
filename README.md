# Kestrel — Vulnerability Intelligence RAG Assistant

Kestrel is a local Retrieval-Augmented Generation (RAG) system for querying cybersecurity vulnerability data. It combines the CISA Known Exploited Vulnerabilities (KEV) catalog, the National Vulnerability Database (NVD), and supplementary PDF advisory documents into a single indexed knowledge base, enabling natural language queries over structured and unstructured threat intelligence.

The symbology of Kestrel reflects my interest in cybersecurity.   The Kestrel is a bird of prey, used to symbolize several relevant cybersecurity and national security missions.    This symbology reflects the importance of  improved tools to guide and manage patching and patch management to the cybersecurity industry.   

---
Project Team:  Mike Worden
---

## Features

- **Multi-source ingestion** — indexes KEV JSON, NVD 2.0 JSON, and PDF documents (CISA advisories, vendor bulletins) into a unified ChromaDB vector store
- **Natural language querying** — ask questions in plain English; Kestrel retrieves relevant chunks and generates grounded answers via an OpenRouter-hosted LLM
- **KEV prioritization** — CISA Known Exploited Vulnerabilities are flagged in both the index and retrieval results for immediate identification of high-priority threats
- **Retrieval visualization** — interactive Plotly chart displays cosine similarity scores for returned chunks, color-coded by source type (KEV / NVD / PDF), with full chunk text preview on hover
- **Conversational memory** — the last three turns of conversation history are injected into each LLM call, supporting follow-up questions without repeating context
- **Gradio chat interface** — browser-based chat UI with example queries, source citations, and embedded retrieval distance chart
- **End-to-end validation** — `validate_connection()` exercises the full pipeline (ChromaDB → retrieval → LLM) at startup and reports which layer fails (if any)


---

## Data Sources

### CISA KEV Catalog JSON
The [CISA Known Exploited Vulnerabilities catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) provides a curated list of CVEs confirmed to have been actively exploited. Each record includes CVE ID, affected vendor/product, required remediation action, due date, and ransomware campaign association. 

### National Vulnerability Database (NVD) 2.0 JSON
The [National Vulnerability Database](https://nvd.nist.gov/) records are loaded from local bulk JSON files conforming to the NVD Vulnerability Data API 2.0 schema. Kestrel extracts CVE descriptions, CVSS scores (v4.0 → v3.1 → v3.0 → v2.0 fallback chain), CWE weaknesses, CPE-affected products, and embedded CISA fields (`cisaExploitAdd`, `cisaActionDue`, `cisaRequiredAction`). 

### PDF Documents
- DOD Instruction 8531.01 DoD Vulnerability Management - Establishes policy, assigns responsibilities, and provides procedures for DoD vulnerability management and response to vulnerabilities identified in all software, firmware, and hardware within the DoD information network (DODIN).
- NIST Special Publication 800-40R4 Guide to Enterprise Patch Management Planning - Sets the standard for patch management for the federal government.
- NIST Special Publication 1800-31 Improving Enterprise Patching for General IT Systems - provides practices to improving patch management in enterprise environments.

All PDF extraction includes the following features:
1. **Header detection** — text is split on markdown headers, Title Case section labels, and ALL CAPS headings using regex
2. **Token-aware subdivision** — sections exceeding the 500-token budget are subdivided into overlapping windows using `tiktoken` (cl100k_base encoding), with the section header prepended to each sub-chunk to preserve retrieval context

-
---

## Installation

**Requirements:** Python 3.11+, pip

```bash
git clone https://github.com/MikeWorden/kestrel.git
cd kestrel
pip install -r requirements.txt
```

**Dependencies:**
```
chromadb
openai
gradio
plotly
pypdf
tiktoken
python-dotenv
tqdm
```

**Environment configuration** — create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=openai/gpt-4o-mini
EMBED_MODEL=text-embedding-3-large
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
DEBUG=false
```

---

## Data Setup

Place data files in the configured `docs_dir` (default: `./data`):
kev - docs_dir/kev
nvd - docs_dir/nvd
pdfs - docs_dir/pdf


**Download KEV catalog:**
```bash
curl -o data/known_exploited_vulnerabilities.json \
  https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
```

**Download NVD bulk data** — see [fkie-cad/nvd-json-data-feeds](https://github.com/fkie-cad/nvd-json-data-feeds) for pre-built annual JSON files.

**Download PDF data*
[DOD INSTRUCTION 8531.01 ](https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/853101p.pdf) 
[NIST SP 800-40 Rev. 4](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-40r4.pdf) 
[NIST SP 1800-31](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1800-31.pdf)

---

## Usage

```bash
python main.py 
```

On startup, Kestrel will:
1. Load and index all data sources (skipped if index already exists)
2. Run end-to-end pipeline validation
3. Launch the Gradio interface and open a browser tab

**Example queries:**
- `What Apache vulnerabilities are in the KEV catalog?`
- `Show me critical CVEs with known ransomware use`
- `How should I prioritize patching`
- `Should I generate an inventory for patching`
- `What Microsoft vulnerabilities were added to KEV in 2024?`

---



## Limitations

- **No cross-session memory** — conversation history is maintained within a browser session only; it is not persisted across restarts
- **Static index** — the index is built once at startup; live CVE feeds require a manual rebuild via `build_index(force_rebuild=True)`
- **Embedding model dependency** — query and document embeddings must use the same model; changing `embed_model` requires a full index rebuild
- **OpenRouter dependency** — LLM generation requires a valid OpenRouter API key and an active internet connection; retrieval operates fully locally

---

## License

MIT
