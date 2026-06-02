"""
config.py - Project Configuration
==================================
Central configuration for the project. S
Usage:
    from config import Config
    cfg = Config()
    print(cfg.LOG_LEVEL)
"""

import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI  
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and load .env
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Application configuration. Override any value via environment variable."""


    def __init__(self):
        # -----------------------------------------------------------------------
        # Application identity
        # -----------------------------------------------------------------------
        self.app_name: str = "Kestrel"
        self.version: str = "0.1.0"

        # -----------------------------------------------------------------------
        # Paths
        # -----------------------------------------------------------------------
        self.base_dir: Path = Path(".")
        self.data_dir: Path = self.base_dir / "data"
        self.log_dir: Path = self.base_dir / "logs"
        self.output_dir: Path = self.base_dir / "output"
        self.kev_dir: Path = self.data_dir / "kev"
        self.nvd_dir: Path = self.data_dir / "nvd"
        self.pdf_dir: Path = self.data_dir / "pdfs"

        # Create directories if they don't exist
        for dir in [self.data_dir, self.log_dir, self.output_dir ]:
            dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------------------
        # External API settings
        # -----------------------------------------------------------------------
        load_dotenv(find_dotenv(usecwd=True))

        # openrouter key for LLM calls
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
       
        # But the embedding function requires an OpenAI key
        if self.openrouter_api_key:
            self.client = OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            print("ERROR: OPENROUTER_API_KEY not found in environment.")

        
            
        # -----------------------------------------------------------------------
        # RAG  settings 
        # -----------------------------------------------------------------------
        
        self.__docs_dir = "./openai_kestrel_docs"  # directory containing source documents (e.g., KEV and NVD JSON files)
        self.chroma_dir = "./chroma_db"
        self.collection_name = "openai_kestrel" 
        self.embed_model = "openai/text-embedding-3-small"
        self.llm_model = "gpt-4o-mini"  
        
        self.chunk_size    = 500
        self.chunk_overlap = 50
        self.top_k         = 10
        self.system_prompt = ('You are a helpful assistant that answers questions about \''
                              'patch management and software, and vulnerabilities, specifically'
                              'Known Exploited Vulnerabilities (KEV) and National Vulnerability Database (NVD)'
                              ' vulnerabilities . Answer using only the provided context. '
                              'If the answer is not in the context, say so clearly.')
    

        self.embed_model    = "text-embedding-3-small" # this is an openai embedding model
        self.llm_model      = "gpt-4o-mini"
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        
        

 

    def __repr__(self) -> str:
        return (
            f"<Config app={self.app_name!r} version={self.version!r} "

        )


# ---------------------------------------------------------------------------
# Convenience singleton (optional — import directly if you prefer)
# ---------------------------------------------------------------------------
# config = Config()


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = Config()
    print(config)
    print("Configuration settings:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    