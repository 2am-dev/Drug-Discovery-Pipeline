# drug_discovery_pipeline/README.md
# Drug Discovery Hypothesis-to-Report Pipeline

An autonomous, multi-agent system that — given a disease indication or a
biological target — produces a comprehensive **Project Proposal Report**
covering literature mining, target selection, in-silico molecule design,
docking, synthetic-accessibility assessment, and retrosynthetic route
proposal.

**All computation runs locally** against an Ollama server (tested with an
NVIDIA 5070 Ti).

---

## Architecture
main.py
└─ PlannerAgent (orchestrator)
├─ RetrieverAgent → literature_search, patent_search
├─ HypothesisAgent → target_lookup, LLM ranking
├─ MoleculeDesignerAgent → molecule_generator (RDKit)
├─ DockingEvaluatorAgent → docking (AutoDock Vina / mock)
├─ SynthesisEvaluatorAgent → synthesis_checker, LLM retrosynthesis
└─ ReportCompilerAgent → Markdown report

- **Agents** (`agents/`) – each owns one pipeline phase, with a `run(state) → state` method.
- **Tools** (`tools/`) – thin wrappers around external APIs (UniProt, PubMed, PatentsView) and computational tools (RDKit, Vina).
- **Utils** (`utils/`) – prompts, LLM/embedding helpers, logging.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** running locally with the following models pulled:
   ```bash
   ollama pull nomic-embed-text
   ollama pull gemma4:31b-it-q8_0
   ```
(Adjust LLM_MODEL in config.py if you use a different model.)

3. ***AutoDock Vina*** (optional – the pipeline falls back to mock docking scores):
```Ubuntu/Debian
sudo apt install autodock-vina

# or download from https://github.com/ccsb-scripps/AutoDock-Vina
```