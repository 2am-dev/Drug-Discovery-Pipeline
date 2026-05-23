# drug_discovery_pipeline/README.md
<!-- ==========================================================================
FILE: README.md
ROLE: Project documentation — setup, usage, architecture overview.
========================================================================== -->

# 🧬 End-to-End AI Drug Discovery Pipeline

An autonomous, multi-agent drug discovery system powered by a **local Ollama**
LLM server. Given a disease indication or biological target, the pipeline
produces a comprehensive **Project Proposal Report** covering literature
mining, target selection, hypothesis formulation, de novo molecule design,
docking evaluation, and synthetic route proposal.

---

## Architecture
User Input ──► PlannerAgent (Orchestrator)
│
┌───────┼──────────────────────────────────┐
▼ ▼ ▼ ▼ ▼ ▼
Retriever Hypothesis Molecule Docking Synthesis Report
Agent Agent Designer Eval. Eval. Compiler
│ │ Agent Agent Agent Agent
▼ ▼ │ │ │ │
Tools Ollama Tools Tools Tools outputs/
(PubMed, LLM (RDKit, (Vina/ (SA proposal_*.md
UniProt, PAINS) mock) score,
Patents) LLM)

All agents share a single **state dictionary** passed sequentially.

---

## Prerequisites

### 1. Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull gemma3:27b          # or any large LLM
ollama pull nomic-embed-text    # embeddings

# Verify
ollama list
```
### 2. Python Environment
``` bash
# Using conda (recommended for RDKit)
conda create -n drugdisco python=3.11 -y
conda activate drugdisco

# Install RDKit via conda-forge
conda install -c conda-forge rdkit openbabel -y

# Install Python dependencies
pip install -r requirements.txt
```
### 3. AutoDock Vina (Optional but recommended)

``` bash
# Ubuntu/Debian
sudo apt install autodock-vina

# Or download from https://vina.scripps.edu/downloads/
# Ensure 'vina' is in your PATH
which vina

# Set path in environment if non-standard
export VINA_PATH=/path/to/vina
```

### 4. Receptor PDBQT (Optional)
If you have a pre-prepared receptor PDBQT file:
``` bash
export RECEPTOR_PDBQT=/path/to/receptor.pdbqt
```
Otherwise the pipeline will download the PDB file from RCSB and attempt
automatic preparation (requires MGLTools or obabel).