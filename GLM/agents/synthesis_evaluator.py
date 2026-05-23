# drug_discovery_pipeline/agents/synthesis_evaluator.py
"""
Synthesis Evaluator Agent – assesses synthetic accessibility and proposes
retrosynthetic routes for the top candidates.
"""

from __future__ import annotations

import logging
from typing import Dict

from tools.synthesis_checker import batch_evaluate

logger = logging.getLogger(__name__)


class SynthesisEvaluatorAgent:
    """Evaluates synthetic accessibility and proposes routes."""

    def run(self, state: Dict) -> Dict:
        logger.info("SynthesisEvaluatorAgent starting…")

        docking_results = state.get("docking_results", [])
        hypothesis = state.get("hypothesis", {})
        target = hypothesis.get("target", state.get("input", ""))
        hypothesis_text = hypothesis.get("hypothesis", "")

        if not docking_results:
            logger.warning("No docking results – nothing to evaluate.")
            state["synthesis_results"] = []
            return state

        # Take top candidates from docking (already sorted by affinity)
        smiles_list = [r["smiles"] for r in docking_results]

        synthesis = batch_evaluate(smiles_list, target=target, hypothesis=hypothesis_text)

        # Merge with docking data
        for s in synthesis:
            for d in docking_results:
                if s["smiles"] == d["smiles"]:
                    s["affinity_kcal_mol"] = d.get("affinity_kcal_mol", 0)
                    s["method"] = d.get("method", "unknown")
                    s["qed"] = d.get("qed", 0)
                    break

        state["synthesis_results"] = synthesis
        logger.info("SynthesisEvaluatorAgent done – %d molecules evaluated.", len(synthesis))
        return state