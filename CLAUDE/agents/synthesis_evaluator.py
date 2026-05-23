# drug_discovery_pipeline/agents/synthesis_evaluator.py
# =============================================================================
# FILE: agents/synthesis_evaluator.py
# ROLE: Synthetic feasibility agent.
#       SA score: RDKit (local)
#       Retrosynthesis LLM call → REMOTE gemma4:31b  (task="retrosynthesis")
#       (routing happens inside tools/synthesis_checker.py)
# =============================================================================

import logging

from tools.synthesis_checker import assess_multiple

log = logging.getLogger("drug_discovery.synthesis_evaluator")


class SynthesisEvaluatorAgent:
    """
    SA score     → RDKit (local, no network)
    Retrosynthesis → REMOTE gemma4:31b (via synthesis_checker → routed_llm_call)
    """

    def run(self, state: dict) -> dict:
        log.info("SynthesisEvaluatorAgent running...")
        dr     = state.get("docking_results", [])
        target = state.get("hypothesis", {}).get("selected_target", {})

        if not dr:
            log.warning("No docking results for synthesis assessment.")
            state["synthesis"] = []
            return state

        smiles_top = [r["smiles"] for r in dr[:3]]
        results    = assess_multiple(smiles_top, target, use_llm=True)
        state["synthesis"] = results

        for r in results:
            log.info(
                f"SA {r.get('sa_score','?')} ({r.get('feasibility','?')}) | "
                f"steps ~{r.get('estimated_steps','?')} | "
                f"{r.get('smiles','')[:50]}"
            )
        return state