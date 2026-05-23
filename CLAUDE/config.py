# drug_discovery_pipeline/config.py
# =============================================================================
# FILE: config.py
# ROLE: Central configuration for the entire pipeline.
#       All model names, API endpoints, file paths, and thresholds live here.
#       Import this module in every other file – never hard-code values.
# =============================================================================

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & output directory
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Temporary working directory for docking files (PDB, PDBQT, etc.)
WORK_DIR = BASE_DIR / "workdir"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB persistence directory
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama / OpenAI-compatible API settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY: str = "ollama"  # Ollama ignores the key but openai client needs it

# Model identifiers (change to any model pulled in your Ollama instance)
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma3:27b")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# LLM generation parameters
LLM_TEMPERATURE: float = 0.3
LLM_MAX_TOKENS: int = 4096
LLM_TIMEOUT: int = 300  # seconds – large models can be slow

# ---------------------------------------------------------------------------
# ChromaDB vector store settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_LITERATURE: str = "drug_discovery_literature"
CHROMA_COLLECTION_PATENTS: str = "drug_discovery_patents"
CHROMA_TOP_K: int = 10          # number of nearest neighbours to retrieve
CHROMA_DISTANCE_THRESHOLD: float = 1.5  # cosine distance cut-off

# ---------------------------------------------------------------------------
# External API endpoints (no auth keys needed for free tiers)
# ---------------------------------------------------------------------------
PUBMED_ENTREZ_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_MAX_RESULTS: int = 20

PATENTSVIEW_BASE: str = "https://search.patentsview.org/api/v1"
PATENTS_MAX_RESULTS: int = 10

UNIPROT_BASE: str = "https://rest.uniprot.org/uniprotkb"
PDB_BASE: str = "https://files.rcsb.org/download"
PDB_SEARCH_BASE: str = "https://search.rcsb.org/rcsbsearch/v2/query"

# ---------------------------------------------------------------------------
# Molecule design thresholds (Lipinski + drug-likeness)
# ---------------------------------------------------------------------------
LIPINSKI_MW_MAX: float = 500.0
LIPINSKI_LOGP_MAX: float = 5.0
LIPINSKI_HBD_MAX: int = 5
LIPINSKI_HBA_MAX: int = 10
QED_MIN: float = 0.4            # Quantitative Estimate of Drug-likeness floor
SA_SCORE_MAX: float = 6.0       # Synthetic Accessibility score ceiling (1=easy,10=hard)
NUM_CANDIDATES_GENERATE: int = 20
NUM_CANDIDATES_SHORTLIST: int = 5

# ---------------------------------------------------------------------------
# Docking settings
# ---------------------------------------------------------------------------
VINA_EXECUTABLE: str = os.getenv("VINA_PATH", "vina")   # must be in PATH
VINA_EXHAUSTIVENESS: int = 8
VINA_NUM_MODES: int = 9
VINA_ENERGY_RANGE: float = 3.0

# User-supplied receptor PDBQT (override via env var or place file in workdir)
# If left empty the pipeline will attempt to download & prepare from PDB.
RECEPTOR_PDBQT_PATH: str = os.getenv("RECEPTOR_PDBQT", "")

# Docking box defaults – override per-run in state dict if known
DEFAULT_BOX_CENTER: tuple = (0.0, 0.0, 0.0)
DEFAULT_BOX_SIZE: tuple = (25.0, 25.0, 25.0)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = OUTPUT_DIR / "pipeline.log"