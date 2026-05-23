"""
Central configuration for the drug discovery pipeline.
All paths, model names, API endpoints, and parameters are defined here.
"""
import os
from pathlib import Path

# ----------------- Ollama Server -----------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4:31b-it-q8_0"  # change to any model available on your Ollama instance

# ----------------- ChromaDB (local vector store) -----------------
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "pubmed_abstracts"

# ----------------- PubMed (Entrez API) -----------------
PUBMED_EMAIL = "your-email@example.com"   # NCBI requires an email address
PUBMED_MAX_RESULTS = 20

# ----------------- Patent Search (PatentsView) -----------------
# PatentsView does not require an API key
PATENT_API_KEY = None

# ----------------- Molecular Docking -----------------
# User must provide a prepared receptor PDBQT file.
RECEPTOR_PDBQT_PATH = "./data/receptor.pdbqt"
# Executable path for AutoDock Vina (must be in PATH or full path)
VINA_PATH = "vina"
# Docking box: center_x, center_y, center_z, size_x, size_y, size_z
VINA_SEARCH_SPACE = [0, 0, 0, 20, 20, 20]

# ----------------- Output Directory -----------------
OUTPUT_DIR = "./outputs"

# ----------------- Logging -----------------
LOG_LEVEL = "INFO"   # DEBUG, INFO, WARNING, ERROR