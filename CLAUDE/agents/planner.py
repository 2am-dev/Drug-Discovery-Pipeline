# drug_discovery_pipeline/agents/planner.py
# =============================================================================
# FILE: agents/planner.py
# ROLE: Orchestrator agent.
#       1. Receives the user input (disease or target).
#       2. Uses the LLM to decompose the task into research phases.
#       3. Initialises the shared pipeline state dict.
#       4. Calls all downstream agents in sequence.
#       5. Returns the final state (including the compiled report).
# =============================================================================

import logging
from datetime import datetime
from typing import Any

from utils.helpers import llm_call, run_id, safe_extract_json, timestamp
from utils.prompts import planner_prompt

log = logging.getLogger("drug_discovery.planner")


class PlannerAgent:
    """
    Orchestrator that decomposes a drug discovery task into phases and
    delegates to specialised agents.

    Usage:
        agent = PlannerAgent()
        state = agent.run("Alzheimer's disease")
    """

    def __init__(self):
        # Import agents here to avoid circular imports at module load time
        from agents.retriever import RetrieverAgent
        from agents.hypothesis import HypothesisAgent
        from agents.molecule_designer import MoleculeDesignerAgent
        from agents.docking_evaluator import DockingEvaluatorAgent
        from agents.synthesis_evaluator import SynthesisEvaluatorAgent
        from agents.report_compiler import ReportCompilerAgent

        self.retriever = RetrieverAgent()
        self.hypothesis = HypothesisAgent()
        self.molecule_designer = MoleculeDesignerAgent()
        self.docking_evaluator = DockingEvaluatorAgent()
        self.synthesis_evaluator = SynthesisEvaluatorAgent()
        self.report_compiler = ReportCompilerAgent()

        log.info("PlannerAgent initialised with all sub-agents.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self, user_input: str) -> dict:
        """Create and return the initial shared state dictionary."""
        rid = run_id()
        return {
            "run_id": rid,
            "input": user_input,
            "started_at": timestamp(),
            "plan": [],
            "literature": {},
            "patents": {},
            "target_info": {},
            "hypothesis": {},
            "pharmacophore": {},
            "candidates": [],
            "docking_results": [],
            "synthesis": [],
            "report": "",
            "errors": [],
            "metadata": {
                "run_id": rid,
                "model": "configured_in_config.py",
            },
        }

    def _decompose_task(self, user_input: str) -> list[dict]:
        """
        Call the LLM to generate a structured research plan.
        Returns a list of phase dicts.
        """
        log.info("Decomposing task with LLM planner...")
        try:
            messages = planner_prompt(user_input)
            raw = llm_call(messages)
            plan_data = safe_extract_json(raw) or {}
            phases = plan_data.get("phases", [])
            if phases:
                log.info(f"Plan contains {len(phases)} phases.")
                for ph in phases:
                    log.info(
                        f"  Phase {ph.get('phase_number')}: {ph.get('phase_name')}"
                    )
            else:
                log.warning("LLM returned no phases; using default plan.")
                phases = self._default_plan()
            return phases
        except Exception as e:
            log.error(f"Task decomposition failed: {e}")
            return self._default_plan()

    def _default_plan(self) -> list[dict]:
        """Fallback plan if LLM fails."""
        return [
            {"phase_number": 1, "phase_name": "Literature & Patent Mining", "description": "Retrieve scientific evidence"},
            {"phase_number": 2, "phase_name": "Target Selection & Hypothesis", "description": "Identify druggable target"},
            {"phase_number": 3, "phase_name": "Molecule Design", "description": "Generate SMILES candidates"},
            {"phase_number": 4, "phase_name": "Docking Evaluation", "description": "Score binding affinity"},
            {"phase_number": 5, "phase_name": "Synthesis Assessment", "description": "Evaluate synthetic routes"},
            {"phase_number": 6, "phase_name": "Report Compilation", "description": "Produce project proposal"},
        ]

    def _log_phase(self, phase_name: str, state: dict) -> None:
        log.info(f"{'='*60}")
        log.info(f"  PHASE: {phase_name}")
        log.info(f"{'='*60}")

    def _record_error(self, state: dict, agent_name: str, error: Exception) -> None:
        """Append a non-fatal error to state without crashing the pipeline."""
        msg = f"{agent_name}: {type(error).__name__}: {error}"
        state["errors"].append({"agent": agent_name, "error": msg, "time": timestamp()})
        log.error(msg)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> dict:
        """
        Execute the complete drug discovery pipeline.

        Args:
            user_input: Disease name, target gene, or therapeutic area.

        Returns:
            The final shared state dict containing all results and the report.
        """
        log.info(f"Pipeline starting for: '{user_input}'")
        state = self._init_state(user_input)

        # ── Phase 0: Planning ─────────────────────────────────────────
        self._log_phase("Task Decomposition", state)
        state["plan"] = self._decompose_task(user_input)

        # ── Phase 1: Literature & Patent Retrieval ────────────────────
        self._log_phase("Literature & Patent Mining", state)
        try:
            state = self.retriever.run(state)
        except Exception as e:
            self._record_error(state, "RetrieverAgent", e)

        # ── Phase 2: Hypothesis Formation ─────────────────────────────
        self._log_phase("Target Selection & Hypothesis", state)
        try:
            state = self.hypothesis.run(state)
        except Exception as e:
            self._record_error(state, "HypothesisAgent", e)

        # ── Phase 3: Molecule Design ───────────────────────────────────
        self._log_phase("Molecule Design", state)
        try:
            state = self.molecule_designer.run(state)
        except Exception as e:
            self._record_error(state, "MoleculeDesignerAgent", e)

        # ── Phase 4: Docking Evaluation ───────────────────────────────
        self._log_phase("Docking Evaluation", state)
        try:
            state = self.docking_evaluator.run(state)
        except Exception as e:
            self._record_error(state, "DockingEvaluatorAgent", e)

        # ── Phase 5: Synthesis Assessment ─────────────────────────────
        self._log_phase("Synthesis Assessment", state)
        try:
            state = self.synthesis_evaluator.run(state)
        except Exception as e:
            self._record_error(state, "SynthesisEvaluatorAgent", e)

        # ── Phase 6: Report Compilation ───────────────────────────────
        self._log_phase("Report Compilation", state)
        try:
            state = self.report_compiler.run(state)
        except Exception as e:
            self._record_error(state, "ReportCompilerAgent", e)

        state["completed_at"] = timestamp()
        log.info(
            f"Pipeline completed. Run ID: {state['run_id']}. "
            f"Errors: {len(state['errors'])}."
        )
        return state