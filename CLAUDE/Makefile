# drug_discovery_pipeline/Makefile
# =============================================================================
# FILE: Makefile
# ROLE: Developer convenience commands.
#       Run `make help` to see all available targets.
# =============================================================================

.PHONY: help install install-dev env-setup lint format type-check test \
        test-fast test-cov clean clean-outputs run run-debug \
        check-servers pre-commit-install ollama-status

# ---------------------------------------------------------------------------
# Colours for terminal output
# ---------------------------------------------------------------------------
CYAN  := \033[0;36m
GREEN := \033[0;32m
YELLOW:= \033[0;33m
RED   := \033[0;31m
RESET := \033[0m

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PYTHON     ?= python3
PIP        ?= pip
QUERY      ?= "Alzheimer's disease"
LOG_LEVEL  ?= INFO

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)Drug Discovery Pipeline — Developer Commands$(RESET)"
	@echo "============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install dev dependencies (includes linters, pytest)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install \
		black isort ruff mypy \
		pytest pytest-cov pytest-mock \
		pre-commit python-dotenv \
		types-requests

env-setup: ## Copy .env.example to .env (if .env doesn't exist)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env from .env.example — please fill in your values!$(RESET)"; \
	else \
		echo "$(GREEN).env already exists — skipping$(RESET)"; \
	fi

dirs: ## Create required runtime directories
	@mkdir -p outputs workdir chroma_db
	@touch outputs/.gitkeep workdir/.gitkeep chroma_db/.gitkeep
	@echo "$(GREEN)Directories created$(RESET)"

pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
format: ## Auto-format with black + isort
	black .
	isort .

lint: ## Run ruff linter
	ruff check . --fix

type-check: ## Run mypy type checker
	mypy agents/ tools/ utils/ config.py main.py \
		--ignore-missing-imports --no-strict-optional || true

check: format lint type-check ## Run all code quality tools

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run all tests (skips Ollama/Vina/network tests)
	pytest tests/ -v \
		-m "not requires_ollama and not requires_vina and not requires_network" \
		--tb=short

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v \
		-m "not requires_ollama and not requires_vina and not requires_network" \
		--tb=short --no-cov

test-cov: ## Run tests with HTML coverage report
	pytest tests/ \
		-m "not requires_ollama and not requires_vina and not requires_network" \
		--cov=. \
		--cov-omit="tests/*,venv/*,.venv/*,*/__pycache__/*" \
		--cov-report=html:htmlcov \
		--cov-report=term-missing
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

test-all: ## Run ALL tests including those needing Ollama/Vina/network
	pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
run: ## Run the pipeline (set QUERY=... to override)
	$(PYTHON) main.py $(QUERY) --log-level $(LOG_LEVEL)

run-debug: ## Run with DEBUG logging
	$(PYTHON) main.py $(QUERY) --log-level DEBUG

run-save: ## Run and save full JSON state
	$(PYTHON) main.py $(QUERY) --save-state --log-level $(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Server utilities
# ---------------------------------------------------------------------------
check-servers: ## Check which Ollama servers are reachable
	@echo "$(CYAN)Checking server status...$(RESET)"
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from utils.helpers import server_is_healthy; \
from config import LOCAL_OLLAMA_URL, REMOTE_OLLAMA_URL; \
local  = server_is_healthy(LOCAL_OLLAMA_URL); \
remote = server_is_healthy(REMOTE_OLLAMA_URL); \
print(f'LOCAL  (5070 Ti): {\"✓ UP\" if local  else \"✗ DOWN\"} — {LOCAL_OLLAMA_URL}'); \
print(f'REMOTE (A6000x3): {\"✓ UP\" if remote else \"✗ DOWN\"} — {REMOTE_OLLAMA_URL}'); \
"

ollama-status: ## Show models on both Ollama instances
	@echo "$(CYAN)LOCAL models:$(RESET)"
	@ollama list 2>/dev/null || echo "  ollama not in PATH or not running"
	@echo ""
	@echo "$(CYAN)REMOTE models:$(RESET)"
	@ollama-remote list 2>/dev/null || echo "  ollama-remote not available"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove Python cache, coverage, and build artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"       -delete 2>/dev/null || true
	find . -type f -name "*.pyo"       -delete 2>/dev/null || true
	find . -type f -name ".coverage"   -delete 2>/dev/null || true
	rm -rf htmlcov/ .pytest_cache/ dist/ build/ *.egg-info/ 2>/dev/null || true
	@echo "$(GREEN)Cache cleared$(RESET)"

clean-outputs: ## Remove generated reports and working files
	rm -f outputs/proposal_*.md outputs/state_*.json outputs/pipeline.log
	rm -f workdir/*.pdb workdir/*.pdbqt workdir/*.sdf
	@echo "$(YELLOW)Outputs and working files cleared$(RESET)"

clean-chroma: ## ⚠ Delete ChromaDB vector store (forces re-embedding)
	rm -rf chroma_db/
	mkdir -p chroma_db
	touch chroma_db/.gitkeep
	@echo "$(RED)ChromaDB deleted — will be rebuilt on next run$(RESET)"

clean-all: clean clean-outputs clean-chroma ## Remove everything generated