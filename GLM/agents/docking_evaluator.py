# drug_discovery_pipeline/agents/docking_evaluator.py
"""
Docking Evaluator Agent – runs molecular docking on shortlisted candidates.
"""

from __future__ import annotations

import logging
from typing import Dict

from tools.docking import dock_molecules

logger = logging.getLogger(__name__)


class DockingEvaluatorAgent:
    """Evaluates binding affinity of shortlisted molecules via docking."""

    def run(self, state: Dict) -> Dict:
        logger.info("DockingEvaluatorAgent starting…")

        shortlisted = state.get("shortlisted_molecules", [])
        if not shortlisted:
            logger.warning("No shortlisted molecules to dock – skipping.")
            state["docking_results"] = []
            return state

        smiles_list = [m["smiles"] for m in shortlisted]

        # Receptor PDBQT: from state (if user set it) or config default
        receptor = state.get("receptor_pdbqt", "")

        results = dock_molecules(smiles_list, receptor_pdbqt=receptor or None)

        # Merge docking results with molecule metadata
        for r in results:
            for m in shortlisted:
                if m["smiles"] == r["smiles"]:
                    r["qed"] = m.get("qed", 0)
                    r["sa_score"] = m.get("sa_score", 0)
                    r["composite_score"] = m.get("composite_score", 0)
                    break

        state["docking_results"] = results
        logger.info("DockingEvaluatorAgent done – %d results.", len(results))
        return state