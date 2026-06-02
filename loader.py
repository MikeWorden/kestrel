from asyncio.log import logger
import json
import logging
from pathlib import Path

import re
from pypdf import PdfReader

from config import Config




class Loader:
    def __init__(self, config):
        self.config = config
        self.docs = []  # store loaded documents here
        self.kev_docs = []
        self.nvd_docs = []
        self.pdf_docs = []
        
       
        
        
    def load_kev_data(self) -> None:
        """Load data from CISA Known Exploited Vulnerabilities (KEV) JSON files. 
        Reads all .json files from config.kev_dir, extracts relevant fields,
        and stores them as a list of dicts in self.kev_docs. """
        
        docs = []
        kev_path = Path(self.config.kev_dir)  

        for filepath in kev_path.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                records = data.get("vulnerabilities", data) \
                        if isinstance(data, dict) else data
                #print (f"Read {len(records)} records from {filepath.name}")
                for record in records:
                    text = self._extract_KEV_text(record)
                    if text:
                        docs.append({
                            "id":       record.get("cveID", f"unknown-{filepath.stem}"),
                            "text":     text,
                            "document_source": record.get("document_source", "kev"),
                            "metadata": {
                                "vendor":       record.get("vendorProject", ""),
                                "product":      record.get("product", ""),
                                "date_added":   record.get("dateAdded", ""),
                                "ransomware":   record.get("knownRansomwareCampaignUse", "Unknown"),
                                "metadata": self._extract_kev_metadata(record),
                            }
                        })

            except (json.JSONDecodeError, OSError) as e:
                print(f"Failed to load {filepath.name}: {e}")
        self.kev_docs = docs
   
    
    
    def _extract_KEV_text(self, record: dict) -> str:
        """
        Serialize a KEV record to a readable string for embedding.
        Structured prose chunks better than raw JSON for semantic search.
        """
        parts = []

        if record.get("cveID"):
            parts.append(f"CVE ID: {record['cveID']}")
        if record.get("vendorProject") and record.get("product"):
            parts.append(f"Affected Product: {record['vendorProject']} {record['product']}")
        if record.get("vulnerabilityName"):
            parts.append(f"Vulnerability: {record['vulnerabilityName']}")
        if record.get("shortDescription"):
            parts.append(f"Description: {record['shortDescription']}")
        if record.get("requiredAction"):
            parts.append(f"Required Action: {record['requiredAction']}")
        if record.get("knownRansomwareCampaignUse"):
            parts.append(f"Ransomware Use: {record['knownRansomwareCampaignUse']}")
        if record.get("dateAdded"):
            parts.append(f"Date Added: {record['dateAdded']}")
        if record.get("dueDate"):
            parts.append(f"Remediation Due: {record['dueDate']}")
        parts.append("document_source: KEV")

        return "\n".join(parts)

    def _extract_kev_metadata(self, record: dict) -> dict:
        """Pull structured fields from KEV record for ChromaDB metadata."""
        return {
            # --- Already have these ---
            "cve_id":       record.get("cveID", ""),
            "vendor":       record.get("vendorProject", ""),
            "product":      record.get("product", ""),
            "vuln_name":    record.get("vulnerabilityName", ""),
            "date_added":   record.get("dateAdded", ""),
            "due_date":     record.get("dueDate", ""),
            "ransomware":   record.get("knownRansomwareCampaignUse", "Unknown"),
            "cwes":         ", ".join(record.get("cwes", [])),   # e.g. "CWE-79, CWE-89"
            "has_notes":    bool(record.get("notes", "")),       # True/False — don't store full notes text
            "document_source": record.get("document_source", "kev"),
            }
   


    def load_nvd_data(self) -> None:
        """
        Read data from the National Vulnerability Database (NVD) via JSON files.
        Reads all JSON files from config.docs_dir and return a flat
        list of normalized document dicts ready for embedding and indexing.

        Each dict has the shape:
        {
            "id":       str,    # CVE ID — used as ChromaDB document ID
            "source":   str,    # "nvd" (for metadata) and "kev" (for KEV status)
            "text":     str,    # prose text for embedding
            "metadata": dict    # structured fields for ChromaDB filtering
        }

        Returns:
            list[dict]: One entry per CVE record. Empty list on failure.
        """
        docs = []

        nvd_path = Path(self.config.nvd_dir)
        json_files = list(nvd_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {self.config.nvd_dir}")
            return docs

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON parse error in {filepath.name}: {e}")
                continue
            except OSError as e:
                print(f"Could not open {filepath.name}: {e}")
                continue

            records = data.get("vulnerabilities", [])
            print(f"Read {len(records)} records from {filepath.name}")

            for item in records:
                cve = item.get("cve", {})

                text = self._extract_nvd_text(cve)
                if not text:
                    print(f"Empty text for {cve.get('id', 'unknown')} — skipping")
                    continue

                docs.append({
                    "id":       cve.get("id"),
                    "text":     text,
                    "record_source":   "nvd",
                    "metadata": self._extract_metadata(cve),
                })
        self.nvd_docs = docs        
        



    def _extract_nvd_text(self, record: dict) -> str:
        """
        Convert a single NVD CVE record to readable prose for embedding.
        Structured text embeds more semantically than raw JSON.

        Args:
            record: A single 'cve' dict from the NVD vulnerabilities array.

        Returns:
            str: Formatted prose string, or empty string if record is unusable.
        """
        parts = []

        # --- Identity ---
        cve_id = record.get("id", "")
        if not cve_id:
            return ""
        parts.append(f"CVE ID: {cve_id}")

        if record.get("vulnStatus"):
            parts.append(f"Status: {record['vulnStatus']}")

        # --- Description (English only) ---
        description = next(
            (d["value"] for d in record.get("descriptions", [])
             if d.get("lang") == "en"),
            ""
        )
        if description:
            parts.append(f"Description: {description}")

        # --- CVSS Score (v4.0 → v3.1 → v3.0 → v2.0) ---
        metrics = record.get("metrics", {})
        for key, ver in [("cvssMetricV40", "4.0"),
                         ("cvssMetricV31", "3.1"),
                         ("cvssMetricV30", "3.0"),
                         ("cvssMetricV2",  "2.0")]:
            entries = metrics.get(key, [])
            entry   = next((e for e in entries if e.get("type") == "Primary"),
                           entries[0] if entries else None)
            if entry:
                cvss     = entry.get("cvssData", {})
                score    = cvss.get("baseScore", "")
                severity = cvss.get("baseSeverity") or entry.get("baseSeverity", "")
                vector   = cvss.get("vectorString", "")
                parts.append(f"CVSS {ver} Score: {score} ({severity})")
                if vector:
                    parts.append(f"Vector: {vector}")
                break  # use highest available version only

        # --- Weaknesses (CWE) ---
        weaknesses = [
            d["value"]
            for w in record.get("weaknesses", [])
            for d in w.get("description", [])
            if d.get("lang") == "en"
        ]
        if weaknesses:
            parts.append(f"Weaknesses (CWE): {', '.join(weaknesses)}")

        # --- Affected Products (CPE) — capped to avoid bloating embeddings ---
        cpe_list = [
            match["criteria"]
            for cfg   in record.get("configurations", [])
            for node  in cfg.get("nodes", [])
            for match in node.get("cpeMatch", [])
            if match.get("vulnerable")
        ]
        if cpe_list:
            parts.append(f"Affected Products (CPE): {'; '.join(cpe_list[:5])}")
            if len(cpe_list) > 5:
                parts.append(f"  ...and {len(cpe_list) - 5} more affected products")

        # --- CISA KEV fields (embedded directly in NVD 2.0 records) ---
        if record.get("cisaExploitAdd"):
            parts.append(f"CISA KEV: Yes")
            if record.get("cisaVulnerabilityName"):
                parts.append(f"KEV Name: {record['cisaVulnerabilityName']}")
            if record.get("cisaRequiredAction"):
                parts.append(f"Required Action: {record['cisaRequiredAction']}")
            parts.append(
                f"KEV Added: {record.get('cisaExploitAdd', '')} | "
                f"Action Due: {record.get('cisaActionDue', '')}"
            )

        # --- Dates ---
        parts.append(f"Published: {record.get('published', '')}")
        parts.append(f"Last Modified: {record.get('lastModified', '')}")

        # --- Document source for filtering (e.g. "nvd" vs "kev") ---
        parts.append(f"document_source: {record.get('document_source', 'nvd')}")
        return "\n".join(parts)

    def _extract_metadata(self, record: dict) -> dict:
        """
        Pull structured fields for ChromaDB metadata storage.
        Enables filtered queries without relying on semantic search.

        Note: ChromaDB metadata values must be str, int, float, or bool.

        Args:
            record: A single 'cve' dict from the NVD vulnerabilities array.

        Returns:
            dict: Flat metadata dict safe for ChromaDB storage.
        """
        # CVSS — same fallback chain as _extract_nvd_text
        metrics     = record.get("metrics", {})
        cvss_score  = 0.0
        cvss_sev    = ""
        cvss_ver    = ""

        for key, ver in [("cvssMetricV40", "4.0"),
                         ("cvssMetricV31", "3.1"),
                         ("cvssMetricV30", "3.0"),
                         ("cvssMetricV2",  "2.0")]:
            entries = metrics.get(key, [])
            entry   = next((e for e in entries if e.get("type") == "Primary"),
                           entries[0] if entries else None)
            if entry:
                cvss       = entry.get("cvssData", {})
                cvss_score = float(cvss.get("baseScore", 0.0))
                cvss_sev   = cvss.get("baseSeverity") or entry.get("baseSeverity", "")
                cvss_ver   = ver
                break

        weaknesses = [
            d["value"]
            for w in record.get("weaknesses", [])
            for d in w.get("description", [])
            if d.get("lang") == "en"
        ]

        return {
            "cve_id":        record.get("id", ""),
            "vuln_status":   record.get("vulnStatus", ""),
            "cvss_score":    cvss_score,
            "cvss_severity": cvss_sev,
            "cvss_version":  cvss_ver,
            "published":     record.get("published", ""),
            "last_modified": record.get("lastModified", ""),
            "weaknesses":    ", ".join(weaknesses),
            "cisa_due":      record.get("cisaActionDue", ""),
            "document_source": record.get("document_source", "nvd"),
        }
    
    
    
    def merge_kev_nvd(self) -> list[dict]:
        """Merge KEV and NVD records on CVE ID into single enriched documents."""
        nvd_index = {d["id"]: d for d in self.nvd_docs}
        merged    = []

        for kev in self.kev_docs:
            cve_id = kev["id"].replace("kev::", "")
            nvd    = nvd_index.get(cve_id)

            if nvd:
                # Combine text from both sources
                combined_text = (
                    f"{nvd['text']}\n\n"
                    f"--- KEV Data ---\n"
                    f"{kev['text']}"
                )
                combined_meta = {
                    **nvd["metadata"],
                    "kev_due":     kev["metadata"].get("due_date", ""),
                    "ransomware":  kev["metadata"].get("ransomware", "Unknown"),
                }
                merged.append({
                    "id":       cve_id,
                    "text":     combined_text,
                    "metadata": combined_meta,
                })
            else:
                # KEV entry with no NVD match — index as-is
                merged.append(kev)

        # Add NVD-only records not in KEV
        kev_ids = {k["id"] for k in self.kev_docs}
        for nvd in self.nvd_docs:
            if nvd["id"] not in kev_ids:
                merged.append(nvd)

        return merged





 
 

 
    def load_pdf_data(self) -> None:
        """
        Read all PDF files from config.docs_dir and return a flat list
        of page-level document dicts.
 
        Each dict has the shape:
        {
            "id":       str,    # "{filename}_p{page_num}"
            "text":     str,    # extracted page text, cleaned
            "source":   "pdf",  # for metadata filtering
            "metadata": dict    # source, page, title, author, doc_type, etc.
        }
 
        Returns:
            list[dict]: One entry per non-blank page. Empty list on failure.
        """
        docs      = []
        pdf_files = list(self.config.pdf_dir.glob("*.pdf"))
 
        if not pdf_files:
            print(f"No PDF files found in {self.docs_dir}")
            return docs
 
        for filepath in pdf_files:
            pages = self._read_pdf_file(filepath)
            docs.extend(pages)
 
        print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s)")
        self.pdf_docs = docs
        
 
    
   
    # Private: loading and cleaning
   
 
    def _read_pdf_file(self, filepath: Path) -> list[dict]:
        """
        Extract and clean text from each page of a PDF file.
 
        Args:
            filepath: Path to the PDF file.
 
        Returns:
            list[dict]: One dict per non-blank page. Empty list on error.
        """
        try:
            reader = PdfReader(str(filepath))
        except Exception as e:
            logger.error(f"Could not open {filepath.name}: {e}")
            return []
 
        # Pull PDF document metadata for enrichment
        meta        = reader.metadata or {}
        title       = meta.get("/Title",   filepath.stem)
        author      = meta.get("/Author",  "")
        subject     = meta.get("/Subject", "")
        total_pages = len(reader.pages)
 
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    f"Could not extract text from {filepath.name} p{page_num}: {e}"
                )
                continue
 
            text = self._clean_text(raw_text)
            if not text:
                logger.debug(f"Skipping blank page: {filepath.name} p{page_num}")
                continue
 
            pages.append({
                "id":   f"{filepath.stem}_p{page_num}",
                "text": text,
                "document_source": "pdf",
                "metadata": {
                    "source":      filepath.name,
                    "title":       title,
                    "author":      author,
                    "subject":     subject,
                    "page":        page_num,
                    "total_pages": total_pages,
                    "document_source":    "pdf",
                }
            })
 
        logger.debug(f"Extracted {len(pages)} pages from {filepath.name}")
        return pages
 
    def _clean_text(self, text: str) -> str:
        """
        Clean raw pypdf extraction output for embedding.
        Handles hyphenated line breaks, collapsed whitespace,
        and effectively blank pages.
 
        Args:
            text: Raw text from pypdf page.extract_text()
 
        Returns:
            str: Cleaned text, or empty string if nothing usable remains.
        """
        # Rejoin hyphenated line breaks ("vulnera-\nbility" → "vulnerability")
        text = re.sub(r"-\n", "", text)
 
        # Collapse multiple whitespace/newlines to single space
        text = re.sub(r"\s+", " ", text)
 
        text = text.strip()
 
        # Discard pages with no substantive content
        if len(text) < 50:
            return ""
 
        return text
 
    
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
            print(f"Debug: {c}")
    
def main():
    config = Config()
    if config.client is None:
        print("Error: OpenAI client not initialized. Please set OPENROUTER_API_KEY in your environment.")
        return
    else:
        docs = []
        loader = Loader(config)
        if config.kev_dir:
            print("Loading KEV data...")
            loader.load_kev_data()
            
            print(f"Loaded {len(loader.kev_docs)} KEV documents.")

        if config.nvd_dir:
            print("Loading NVD data...")
            loader.load_nvd_data()
            
            print(f"Loaded {len(loader.nvd_docs)} NVD documents.")
        
        
        docs = loader.merge_kev_nvd()
        print(f"Total documents after merging KEV and NVD: {len(docs)}")
            
        if config.pdf_dir:
            print("Loading PDF data...")
            loader.load_pdf_data()
            docs.extend(loader.pdf_docs)
            print(f"Loaded {len(loader.pdf_docs)} PDF page documents.")

        print(f"Total documents loaded: {len(docs)}")
        for i, doc in enumerate(docs[:5], start=1):
            print(f"\n*************Document {i}*************")
            print(f"ID: {doc['metadata']}") 
            


if __name__ == "__main__":
    main()  
    