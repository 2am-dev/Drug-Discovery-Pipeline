# drug_discovery_pipeline/agents/planner.py
# =============================================================================
# FILE: agents/planner.py
# ROLE: Orchestrator agent.
#       Routes task decomposition to REMOTE gemma4:31b.
#       Calls all downstream agents sequentially.
# =============================================================================

import logging
from datetime import datetime

from utils.helpers import routed_llm_call, run_id, safe_extract_json, timestamp
from utils.prompts import planner_prompt

log = logging.getLogger("drug_discovery.planner")


class PlannerAgent:
    """
    Orchestrator. Decomposes the drug discovery task and runs agents in order.
    Uses REMOTE gemma4:31b for planning (task="plan").
    """

    def __init__(self):
        from agents.retriever import RetrieverAgent
        from agents.hypothesis import HypothesisAgent
        from agents.molecule_designer import MoleculeDesignerAgent
        from agents.docking_evaluator import DockingEvaluatorAgent
        from agents.synthesis_evaluator import SynthesisEvaluatorAgent
        from agents.report_compiler import ReportCompilerAgent

        self.retriever          = RetrieverAgent()
        self.hypothesis         = HypothesisAgent()
        self.molecule_designer  = MoleculeDesignerAgent()
        self.docking_evaluator  = DockingEvaluatorAgent()
        self.synthesis_evaluator = SynthesisEvaluatorAgent()
        self.report_compiler    = ReportCompilerAgent()
        log.info("PlannerAgent initialised (dual-server mode).")

    # ------------------------------------------------------------------

    def _init_state(self, user_input: str) -> dict:
        rid = run_id()
        return {
            "run_id": rid, "input": user_input,
            "started_at": timestamp(), "plan": [],
            "literature": {}, "patents": {}, "target_info": {},
            "hypothesis": {}, "pharmacophore": {},
            "candidates": {}, "docking_results": [],
            "docking_interpretation": {},
            "synthesis": [], "report": "", "errors": [],
            "metadata": {"run_id": rid},
        }

    def _decompose_task(self, user_input: str) -> list[dict]:
        log.info("Decomposing task → REMOTE gemma4:31b ...")
        try:
            messages = planner_prompt(user_input)
            # ← routed_llm_call uses task="plan" → REMOTE gemma4:31b
            raw = routed_llm_call("plan", messages)
            data = safe_extract_json(raw) or {}
            phases = data.get("phases", [])
            if phases:
                log.info(f"Plan: {len(phases)} phases")
                return phases
        except Exception as e:
            log.error(f"Task decomposition failed: {e}")
        return self._default_plan()

    def _default_plan(self) -> list[dict]:
        return [
            {"phase_number": i, "phase_name": n, "description": d}
            for i, n, d in [
                (1, "Literature & Patent Mining",     "Retrieve evidence"),
                (2, "Target Selection & Hypothesis",  "Identify druggable target"),
                (3, "Molecule Design",                "Generate SMILES"),
                (4, "Docking Evaluation",             "Score binding affinity"),
                (5, "Synthesis Assessment",           "Evaluate routes"),
                (6, "Report Compilation",             "Produce proposal"),
            ]
        ]

    def _log_phase(self, name: str) -> None:
        log.info(f"{'='*60}\n  PHASE: {name}\n{'='*60}")

    def _record_error(self, state: dict, agent: str, exc: Exception) -> None:
        msg = f"{agent}: {type(exc).__name__}: {exc}"
        state["errors"].append({"agent": agent, "error": msg, "time": timestamp()})
        log.error(msg)

    def run(self, user_input: str) -> dict:
        log.info(f"Pipeline starting: '{user_input}'")
        state = self._init_state(user_input)

        self._log_phase("Task Decomposition")
        state["plan"] = self._decompose_task(user_input)

        for phase_name, agent_name, agent in [
            ("Literature & Patent Mining",    "RetrieverAgent",           self.retriever),
            ("Target Selection & Hypothesis", "HypothesisAgent",          self.hypothesis),
            ("Molecule Design",               "MoleculeDesignerAgent",    self.molecule_designer),
            ("Docking Evaluation",            "DockingEvaluatorAgent",    self.docking_evaluator),
            ("Synthesis Assessment",          "SynthesisEvaluatorAgent",  self.synthesis_evaluator),
            ("Report Compilation",            "ReportCompilerAgent",      self.report_compiler),
        ]:
            self._log_phase(phase_name)
            try:
                state = agent.run(state)
            except Exception as e:
                self._record_error(state, agent_name, e)

        state["completed_at"] = timestamp()
        log.info(
            f"Pipeline done. Run: {state['run_id']}. "
            f"Errors: {len(state['errors'])}."
        )
        return state