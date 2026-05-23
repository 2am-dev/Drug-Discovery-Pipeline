# drug_discovery_pipeline/agents/planner.py
"""
Planner Agent – orchestrates the entire pipeline.

1. Uses the LLM to analyse the user input and produce an execution plan.
2. Calls each downstream agent in order, passing a shared state dict.
3. Returns the final state containing the compiled report.
"""

from __future__ import annotations

import logging
from typing import Dict

from utils.helpers import call_llm
from utils.prompts import PLAN_SYSTEM, PLAN_PROMPT
from agents.retriever import RetrieverAgent
from agents.hypothesis import HypothesisAgent
from agents.molecule_designer import MoleculeDesignerAgent
from agents.docking_evaluator import DockingEvaluatorAgent
from agents.synthesis_evaluator import SynthesisEvaluatorAgent
from agents.report_compiler import ReportCompilerAgent

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Master orchestrator.  Runs the pipeline end-to-end:
        Plan → Retrieve → Hypothesise → Design → Dock → Evaluate → Report
    """

    def __init__(self) -> None:
        self.retriever = RetrieverAgent()
        self.hypothesis = HypothesisAgent()
        self.designer = MoleculeDesignerAgent()
        self.docker = DockingEvaluatorAgent()
        self.synth_eval = SynthesisEvaluatorAgent()
        self.compiler = ReportCompilerAgent()

    def run(self, state: Dict) -> Dict:
        user_input = state.get("input", "")
        logger.info("═══ Pipeline started for: '%s' ═══", user_input)

        # ── Phase 0: Planning ────────────────────────────────────────────
        logger.info("Phase 0 – Planning…")
        plan_prompt = PLAN_PROMPT.format(input=user_input)
        plan = call_llm(plan_prompt, system=PLAN_SYSTEM, temperature=0.3, max_tokens=512)
        state["plan"] = plan
        logger.info("Plan:\n%s", plan)

        # Detect input type from plan text
        plan_lower = plan.lower()
        if "target" in plan_lower.split("input appears to be")[-1][:50] if "input appears to be" in plan_lower else False:
            state["input_type"] = "target"
        else:
            state["input_type"] = "disease"

        # ── Phase 1: Literature & Patent Mining ──────────────────────────
        logger.info("Phase 1 – Literature & Patent Mining…")
        state = self.retriever.run(state)

        # ── Phase 2: Target Selection & Hypothesis ───────────────────────
        logger.info("Phase 2 – Target Selection & Hypothesis…")
        state = self.hypothesis.run(state)

        # ── Phase 3: Molecule Design ─────────────────────────────────────
        logger.info("Phase 3 – Molecule Design…")
        state = self.designer.run(state)

        # ── Phase 4: Docking ─────────────────────────────────────────────
        logger.info("Phase 4 – Docking & Binding-Affinity Prediction…")
        state = self.docker.run(state)

        # ── Phase 5: Synthesis Evaluation ────────────────────────────────
        logger.info("Phase 5 – Synthetic Accessibility & Route Proposal…")
        state = self.synth_eval.run(state)

        # ── Phase 6: Report Compilation ──────────────────────────────────
        logger.info("Phase 6 – Report Compilation…")
        state = self.compiler.run(state)

        logger.info("═══ Pipeline complete ═══")
        return state