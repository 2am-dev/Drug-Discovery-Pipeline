# drug_discovery_pipeline/scripts/check_environment.sh
#!/usr/bin/env bash
# =============================================================================
# FILE: scripts/check_environment.sh
# ROLE: Pre-flight environment checker.
#       Verifies Python version, required packages, Ollama servers,
#       and optional tools (Vina, obabel).
#       Run before your first pipeline execution.
#
# Usage:
#   chmod +x scripts/check_environment.sh
#   ./scripts/check_environment.sh
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; FAILED=1; }

FAILED=0

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   Drug Discovery Pipeline — Environment Check              ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Python version ────────────────────────────────────────────────────────────
echo -e "${CYAN}[1] Python${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    ok "Python $PYTHON_VERSION (>=3.10 required)"
else
    fail "Python $PYTHON_VERSION — need 3.10+"
fi

# ── .env file ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[2] Configuration${NC}"
if [ -f ".env" ]; then
    ok ".env file found"
    if grep -q "REPLACE_WITH_REMOTE_HOST" .env 2>/dev/null; then
        warn "REMOTE_OLLAMA_URL still has placeholder — update .env!"
    fi
else
    fail ".env not found — run: cp .env.example .env"
fi

# ── Required Python packages ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[3] Python Packages${NC}"
PACKAGES=("openai" "chromadb" "requests" "bs4" "feedparser" "rdkit" "pandas" "pydantic" "dotenv")
for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ok "$pkg"
    else
        fail "$pkg — run: pip install -r requirements.txt"
    fi
done

# ── Ollama LOCAL ──────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4] Ollama Servers${NC}"
LOCAL_URL="${LOCAL_OLLAMA_URL:-http://localhost:11434}"

if curl -s --max-time 5 "$LOCAL_URL/api/tags" > /dev/null 2>&1; then
    ok "LOCAL Ollama reachable ($LOCAL_URL)"
    # Check for nomic-embed-text
    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        ok "  nomic-embed-text model present"
    else
        warn "  nomic-embed-text not found — run: ollama pull nomic-embed-text"
    fi
else
    fail "LOCAL Ollama not reachable ($LOCAL_URL) — is ollama serve running?"
fi

# Ollama REMOTE
REMOTE_URL="${REMOTE_OLLAMA_URL:-http://localhost:11435}"
if curl -s --max-time 8 "${REMOTE_URL%/v1}/api/tags" > /dev/null 2>&1; then
    ok "REMOTE Ollama reachable ($REMOTE_URL)"
    if command -v ollama-remote &>/dev/null; then
        if ollama-remote list 2>/dev/null | grep -q "gemma4:31b"; then
            ok "  gemma4:31b-it-q8_0 present"
        else
            warn "  gemma4:31b-it-q8_0 not found on remote"
        fi
    fi
else
    warn "REMOTE Ollama not reachable ($REMOTE_URL) — pipeline will fall back to local models"
fi

# ── Optional tools ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[5] Optional Tools${NC}"

if command -v vina &>/dev/null; then
    VINA_VER=$(vina --version 2>&1 | head -1)
    ok "AutoDock Vina: $VINA_VER"
else
    warn "AutoDock Vina not found — mock docking scores will be used"
    warn "  Install: https://vina.scripps.edu/downloads/"
fi

if command -v obabel &>/dev/null; then
    BABEL_VER=$(obabel --version 2>&1 | head -1)
    ok "Open Babel: $BABEL_VER"
else
    warn "Open Babel (obabel) not found — PDBQT conversion unavailable"
    warn "  Install: conda install -c conda-forge openbabel"
fi

if python3 -c "import aizynthfinder" 2>/dev/null; then
    ok "AiZynthFinder installed"
else
    warn "AiZynthFinder not installed — using LLM-only retrosynthesis"
    warn "  Install: pip install aizynthfinder (optional)"
fi

# ── Directories ───────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[6] Directories${NC}"
for dir in outputs workdir chroma_db; do
    if [ -d "$dir" ]; then
        ok "$dir/ exists"
    else
        mkdir -p "$dir"
        warn "$dir/ created"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}✓ Environment looks good! Ready to run.${NC}"
    echo ""
    echo -e "  Quick start:"
    echo -e "  ${CYAN}  python main.py \"Alzheimer's disease\"${NC}"
else
    echo -e "  ${RED}✗ Some checks failed — fix the issues above before running.${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

exit $FAILED