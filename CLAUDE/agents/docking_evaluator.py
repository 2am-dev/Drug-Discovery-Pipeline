# drug_discovery_pipeline/agents/docking_evaluator.py
# =============================================================================
# FILE: agents/docking_evaluator.py
# ROLE: Molecular docking evaluation agent.
#       Docks each shortlisted SMILES candidate against the target receptor.
#       Uses AutoDock Vina (or mock scorer) via the docking tool.
#       Calls LLM to interpret results and rank candidates.
# =============================================================================

import logging

from tools.docking import dock_multiple
from utils.helpers import llm_call, safe_extract_json
from utils.prompts import docking_interpretation_prompt

log = logging.getLogger("drug_discovery.docking_evaluator")


class DockingEvaluatorAgent:
    """
    Evaluates binding affinity of shortlisted molecules via docking.

    Expects state keys: candidates, hypothesis, target_info
    Produces state keys: docking_results
    """

    def run(self, state: dict) -> dict:
        log.info("DockingEvaluatorAgent running...")

        shortlist = state.get("candidates", {}).get("shortlist", [])
        if not shortlist:
            log.warning("No shortlisted candidates to dock.")
            state["docking_results"] = []
            return state

        target = state.get("hypothesis", {}).get("selected_target", {})
        pdb_local = state.get("hypothesis", {}).get("pdb_local_path", "")

        smiles_list = [m["smiles"] for m in shortlist]
        log.info(f"Docking {len(smiles_list)} candidates...")

        # ── Run docking ────────────────────────────────────────────────
        raw_results = dock_multiple(
            smiles_list=smiles_list,
            receptor_pdbqt_path=pdb_local or None,
        )

        # ── Merge with property data ───────────────────────────────────
        props_map = {m["smiles"]: m for m in shortlist}
        for r in raw_results:
            mol_props = props_map.get(r["smiles"], {})
            r.update(
                {
                    "QED": mol_props.get("QED", "N/A"),
                    "MW": mol_props.get("MW", "N/A"),
                    "LogP": mol_props.get("LogP", "N/A"),
                    "SA_Score": mol_props.get("SA_Score", "N/A"),
                    "docking_mode": r.get("mode", "unknown"),
                }
            )

        state["docking_results"] = raw_results
        log.info(
            f"Docking complete. Best score: "
            f"{raw_results[0]['score'] if raw_results else 'N/A'} kcal/mol"
        )

        # ── LLM interpretation ─────────────────────────────────────────
        try:
            messages = docking_interpretation_prompt(target, raw_results)
            raw = llm_call(messages)
            interpretation = safe_extract_json(raw)
            if interpretation:
                state["docking_interpretation"] = interpretation
                log.info(
                    f"Best candidate identified: "
                    f"{interpretation.get('best_candidate', 'N/A')[:60]}"
                )
        except Exception as e:
            log.error(f"Docking interpretation LLM call failed: {e}")
            state["docking_interpretation"] = {}

        return state