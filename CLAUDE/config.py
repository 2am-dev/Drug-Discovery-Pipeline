# drug_discovery_pipeline/config.py
# =============================================================================
# FILE: config.py
# ROLE: Central configuration — now with DUAL Ollama server support.
#
# SERVERS:
#   LOCAL  → http://localhost:11434   (RTX 5070 Ti, yours alone)
#   REMOTE → configured via OLLAMA_REMOTE_URL  (3x A6000, shared)
#
# ROUTING PHILOSOPHY:
#   Heavy reasoning  → REMOTE  gemma4:31b-it-q8_0   (best quality)
#   Biomedical NLP   → LOCAL   medgemma1.5           (domain expert)
#   JSON/code tasks  → LOCAL   deepseek-coder:6.7b   (structured output)
#   Embeddings       → REMOTE  nomic-embed-text       (same model, more VRAM)
#   Fallback         → LOCAL   mistral:7b             (always available)
# =============================================================================

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs";   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR   = BASE_DIR / "workdir";   WORK_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR = BASE_DIR / "chroma_db"; CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ── SERVER DEFINITIONS ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# LOCAL Ollama  (RTX 5070 Ti — yours alone)
LOCAL_OLLAMA_URL: str = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434/v1")

# REMOTE Ollama (3× A6000 — shared, no new model pulls)
REMOTE_OLLAMA_URL: str = os.getenv(
    "REMOTE_OLLAMA_URL", "http://localhost:11435/v1"
    # ↑ Replace with the actual remote host, e.g.:
    #   "http://192.168.1.50:11434/v1"
    #   or set env var:  export REMOTE_OLLAMA_URL=http://<host>:11434/v1
)

OLLAMA_API_KEY: str = "ollama"  # Ollama ignores this; openai client needs it

# ---------------------------------------------------------------------------
# ── MODEL ASSIGNMENTS ────────────────────────────────────────────────────────
# Each key maps to (model_name, server_url)
# ---------------------------------------------------------------------------

# PRIMARY heavy-reasoning model — REMOTE A6000 cluster
REMOTE_LLM_MODEL:     str = "gemma4:31b-it-q8_0"

# Biomedical-specialist model — LOCAL (medgemma fine-tuned on medical text)
LOCAL_BIO_MODEL:      str = "medgemma1.5:latest"

# Code / structured-JSON model — LOCAL (best at producing valid JSON/SMILES)
LOCAL_CODE_MODEL:     str = "deepseek-coder:6.7b"

# Lightweight fallback — LOCAL (used when remote is overloaded)
LOCAL_FALLBACK_MODEL: str = "mistral:7b"

# Embeddings — REMOTE preferred (larger VRAM = faster batch embedding)
EMBEDDING_MODEL:      str = "nomic-embed-text"
EMBEDDING_SERVER:     str = REMOTE_OLLAMA_URL   # falls back to LOCAL in helpers

# ---------------------------------------------------------------------------
# Task → (model, server) routing table
# Agents import TASK_MODEL_MAP and call routed_llm_call(task, messages)
# ---------------------------------------------------------------------------
TASK_MODEL_MAP: dict[str, tuple[str, str]] = {
    # ── REMOTE tasks (need deep reasoning or large context) ──────────────
    "plan":               (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "hypothesis":         (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "pharmacophore":      (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "molecule_refine":    (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "docking_interpret":  (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "retrosynthesis":     (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "executive_summary":  (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
    "conclusion":         (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),

    # ── LOCAL biomedical tasks ────────────────────────────────────────────
    "literature_synthesis": (LOCAL_BIO_MODEL, LOCAL_OLLAMA_URL),
    "admet_commentary":     (LOCAL_BIO_MODEL, LOCAL_OLLAMA_URL),
    "target_druggability":  (LOCAL_BIO_MODEL, LOCAL_OLLAMA_URL),

    # ── LOCAL code/JSON tasks ─────────────────────────────────────────────
    "json_repair":          (LOCAL_CODE_MODEL, LOCAL_OLLAMA_URL),
    "smiles_validate":      (LOCAL_CODE_MODEL, LOCAL_OLLAMA_URL),
    "structured_format":    (LOCAL_CODE_MODEL, LOCAL_OLLAMA_URL),

    # ── LOCAL fast/polish tasks ───────────────────────────────────────────
    "section_polish":       (LOCAL_FALLBACK_MODEL, LOCAL_OLLAMA_URL),
    "short_summary":        (LOCAL_FALLBACK_MODEL, LOCAL_OLLAMA_URL),

    # ── Fallback (used by generic llm_call) ──────────────────────────────
    "default":              (REMOTE_LLM_MODEL, REMOTE_OLLAMA_URL),
}

# ---------------------------------------------------------------------------
# LLM generation parameters
# ---------------------------------------------------------------------------
LLM_TEMPERATURE: float = 0.3
LLM_MAX_TOKENS:  int   = 4096
LLM_TIMEOUT:     int   = 360    # seconds — 31b on shared server can queue

# How many times to retry on transient failures
LLM_RETRIES:     int   = 3
LLM_RETRY_DELAY: float = 8.0    # seconds between retries

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_LITERATURE: str   = "drug_discovery_literature"
CHROMA_COLLECTION_PATENTS:    str   = "drug_discovery_patents"
CHROMA_TOP_K:                 int   = 10
CHROMA_DISTANCE_THRESHOLD:    float = 1.5

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------
PUBMED_ENTREZ_BASE:  str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_MAX_RESULTS:  int = 20

PATENTSVIEW_BASE:    str = "https://search.patentsview.org/api/v1"
PATENTS_MAX_RESULTS: int = 10

UNIPROT_BASE:        str = "https://rest.uniprot.org/uniprotkb"
PDB_BASE:            str = "https://files.rcsb.org/download"
PDB_SEARCH_BASE:     str = "https://search.rcsb.org/rcsbsearch/v2/query"

# ---------------------------------------------------------------------------
# Molecule design thresholds
# ---------------------------------------------------------------------------
LIPINSKI_MW_MAX:  float = 500.0
LIPINSKI_LOGP_MAX: float = 5.0
LIPINSKI_HBD_MAX: int   = 5
LIPINSKI_HBA_MAX: int   = 10
QED_MIN:          float = 0.4
SA_SCORE_MAX:     float = 6.0
NUM_CANDIDATES_GENERATE:  int = 20
NUM_CANDIDATES_SHORTLIST: int = 5

# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------
VINA_EXECUTABLE:    str   = os.getenv("VINA_PATH", "vina")
VINA_EXHAUSTIVENESS: int  = 8
VINA_NUM_MODES:     int   = 9
VINA_ENERGY_RANGE:  float = 3.0
RECEPTOR_PDBQT_PATH: str  = os.getenv("RECEPTOR_PDBQT", "")
DEFAULT_BOX_CENTER: tuple = (0.0, 0.0, 0.0)
DEFAULT_BOX_SIZE:   tuple = (25.0, 25.0, 25.0)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str  = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE:  Path = OUTPUT_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Health-check timeout (used by helpers to test server reachability)
# ---------------------------------------------------------------------------
SERVER_HEALTH_TIMEOUT: int = 10  # seconds