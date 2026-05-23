# drug_discovery_pipeline/config.py
"""
Central configuration for the Drug Discovery Pipeline.
All model names, API endpoints, file paths, and tunables live here.
"""

import os

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
LLM_MODEL = "gemma4:31b-it-q8_0"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_TIMEOUT = 180          # seconds – large models can be slow
LLM_MAX_RETRIES = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_store")

# ── Docking ───────────────────────────────────────────────────────────────────
VINA_EXECUTABLE = "vina"        # must be on PATH, or set an absolute path
RECEPTOR_PDBQT = ""             # user-provided .pdbqt; leave empty for mock mode
VINA_CENTER = (0.0, 0.0, 0.0)  # docking box centre
VINA_SIZE = (20, 20, 20)        # docking box dimensions (Å)

# ── External APIs ─────────────────────────────────────────────────────────────
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENTREZ_EMAIL = "drug-pipeline@example.com"   # NCBI requires an email
PATENTSVIEW_BASE = "https://api.patentsview.org/patents/query"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
PDB_DOWNLOAD_BASE = "https://files.rcsb.org/download"

# ── Molecule generation ───────────────────────────────────────────────────────
NUM_CANDIDATES_GENERATE = 20
NUM_CANDIDATES_SHORTLIST = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ── Ensure directories exist ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)