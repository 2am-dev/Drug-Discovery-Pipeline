# drug_discovery_pipeline/scripts/setup_conda_env.sh
#!/usr/bin/env bash
# =============================================================================
# FILE: scripts/setup_conda_env.sh
# ROLE: One-shot conda environment setup script.
#       Creates a fresh conda env with Python 3.11, RDKit, Open Babel,
#       and all pip dependencies.
#
# Usage:
#   chmod +x scripts/setup_conda_env.sh
#   ./scripts/setup_conda_env.sh
#   conda activate drugdisco
# =============================================================================

set -euo pipefail

ENV_NAME="drugdisco"
PYTHON_VERSION="3.11"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Drug Discovery Pipeline — Conda Environment Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo "❌ conda not found. Install Miniconda/Anaconda first."
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing env if requested
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠  Environment '$ENV_NAME' already exists."
    read -rp "   Remove and recreate? [y/N] " answer
    if [[ "${answer,,}" == "y" ]]; then
        conda env remove -n "$ENV_NAME" -y
    else
        echo "   Skipping creation — activating existing env."
        conda activate "$ENV_NAME" 2>/dev/null || \
            echo "   Run: conda activate $ENV_NAME"
        exit 0
    fi
fi

echo "[1/5] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "[2/5] Installing RDKit and Open Babel via conda-forge..."
conda run -n "$ENV_NAME" conda install -c conda-forge \
    rdkit openbabel -y

echo "[3/5] Installing pip dependencies..."
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo "[4/5] Installing dev tools..."
conda run -n "$ENV_NAME" pip install \
    python-dotenv black isort ruff mypy \
    pytest pytest-cov pytest-mock pre-commit

echo "[5/5] Setting up .env..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ⚠  Created .env from .env.example — please fill in REMOTE_OLLAMA_URL!"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. conda activate $ENV_NAME"
echo "  2. Edit .env  →  set REMOTE_OLLAMA_URL=http://<host>:11434/v1"
echo "  3. bash scripts/check_environment.sh"
echo "  4. python main.py \"Alzheimer's disease\""
echo "═══════════════════════════════════════════════════"
echo ""