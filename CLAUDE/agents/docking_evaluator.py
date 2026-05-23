# drug_discovery_pipeline/agents/docking_evaluator.py
# =============================================================================
# FILE: agents/docking_evaluator.py
# ROLE: Molecular docking evaluation agent.
#       Docking itself: RDKit + Vina subprocess (local, no LLM)
#       Result interpretation: REMOTE gemma4:31b  (task="docking_interpret")
# =============================================================================

import logging

from tools.docking import dock_multiple
from utils.helpers import routed_llm_call, safe_extract_json
from utils.prompts import docking_interpretation_prompt

log = logging.getLogger("drug_discovery.docking_evaluator")


class DockingEvaluatorAgent:
    """
    Docking computation  → Vina/mock (local, RDKit)
    Result interpretation → REMOTE gemma4:31b
    """

    def run(self, state: dict) -> dict:
        log.info("DockingEvaluatorAgent running...")
        shortlist  = state.get("candidates", {}).get("shortlist", [])
        if not shortlist:
            log.warning("No shortlisted candidates.")
            state["docking_results"] = []
            return state

        target     = state.get("hypothesis", {}).get("selected_target", {})
        pdb_local  = state.get("hypothesis", {}).get("pdb_local_path", "")
        smi_list   = [m["smiles"] for m in shortlist]

        raw_results = dock_multiple(smi_list, receptor_pdbqt_path=pdb_local or None)

        props_map = {m["smiles"]: m for m in shortlist}
        for r in raw_results:
            mp = props_map.get(r["smiles"], {})
            r.update({
                "QED": mp.get("QED","N/A"), "MW": mp.get("MW","N/A"),
                "LogP": mp.get("LogP","N/A"), "SA_Score": mp.get("SA_Score","N/A"),
                "docking_mode": r.get("mode","unknown"),
            })

        state["docking_results"] = raw_results
        log.info(
            f"Docking done. Best: "
            f"{raw_results[0]['score'] if raw_results else 'N/A'} kcal/mol"
        )

        # ── LLM interpretation (REMOTE) ───────────────────────────────
        try:
            messages = docking_interpretation_prompt(target, raw_results)
            raw = routed_llm_call("docking_interpret", messages)   # ← REMOTE
            interp = safe_extract_json(raw)
            if interp:
                state["docking_interpretation"] = interp
                log.info(f"Best candidate: {interp.get('best_candidate','?')[:60]}")
        except Exception as e:
            log.error(f"Docking interpretation failed: {e}")
            state["docking_interpretation"] = {}

        return state