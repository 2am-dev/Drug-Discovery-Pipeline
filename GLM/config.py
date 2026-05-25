# drug_discovery_pipeline/config.py
"""
Central configuration – now supports TWO Ollama servers.

Strategy:
  - REMOTE server (big GPU)  → heavy LLM reasoning
  - LOCAL  server (small GPU) → embeddings + light LLM tasks
"""

import os

# ── Remote Ollama (big models) ───────────────────────────────────────────────
# Change this to your remote server's URL / port
REMOTE_OLLAMA_BASE_URL = "http://10.10.27.37:11434/v1"
REMOTE_OLLAMA_API_KEY  = "ollama"
REMOTE_LLM_MODEL       = "gemma4:31b-it-q8_0"
# Fallback models on remote (tried in order if primary fails)
REMOTE_LLM_FALLBACKS   = ["gemma4:26b-a4b-it-q8_0", "gpt-oss:20b"]

# ── Local Ollama (smaller models) ────────────────────────────────────────────
LOCAL_OLLAMA_BASE_URL  = "http://localhost:11434/v1"
LOCAL_OLLAMA_API_KEY   = "ollama"
LOCAL_LLM_MODEL        = "llama3:8b"          # or "mistral:7b"
LOCAL_LLM_FALLBACKS    = ["mistral:7b", "phi3:mini"]

# ── Embeddings (run locally to avoid network latency) ────────────────────────
EMBEDDING_BASE_URL     = "http://localhost:11434/v1"   # local server
EMBEDDING_API_KEY      = "ollama"
EMBEDDING_MODEL        = "nomic-embed-text"

# ── Task → server routing ────────────────────────────────────────────────────
# "heavy"  → remote server (planning, hypothesis, retrosynthesis, report)
# "light"  → local server  (search queries, pharmacophore, simple extraction)
TASK_ROUTES = {
    "plan":             "heavy",   # PlannerAgent
    "search_queries":   "light",   # RetrieverAgent – query generation
    "patent_queries":   "light",   # RetrieverAgent – patent query generation
    "hypothesis":       "heavy",   # HypothesisAgent
    "pharmacophore":    "light",   # MoleculeDesignerAgent
    "retrosynthesis":   "heavy",   # SynthesisEvaluatorAgent
    "report":           "heavy",   # ReportCompilerAgent
}

# ── Timeouts / retries ───────────────────────────────────────────────────────
LLM_TIMEOUT      = 300       # seconds – big models on remote can be slow
LLM_MAX_RETRIES  = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR         = os.path.join(PROJECT_ROOT, "outputs")
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_store")

# ── Docking ───────────────────────────────────────────────────────────────────
VINA_EXECUTABLE   = "vina"
RECEPTOR_PDBQT    = ""
VINA_CENTER       = (0.0, 0.0, 0.0)
VINA_SIZE         = (20, 20, 20)

# ── External APIs ─────────────────────────────────────────────────────────────
ENTREZ_BASE       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENTREZ_EMAIL      = "drug-pipeline@example.com"
PATENTSVIEW_BASE  = "https://api.patentsview.org/patents/query"
UNIPROT_BASE      = "https://rest.uniprot.org/uniprotkb"
PDB_DOWNLOAD_BASE = "https://files.rcsb.org/download"

# ── Molecule generation ───────────────────────────────────────────────────────
NUM_CANDIDATES_GENERATE  = 20
NUM_CANDIDATES_SHORTLIST = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ── Ensure directories exist ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)