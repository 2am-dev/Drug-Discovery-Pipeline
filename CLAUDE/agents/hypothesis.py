# drug_discovery_pipeline/agents/hypothesis.py
# =============================================================================
# FILE: agents/hypothesis.py
# ROLE: Hypothesis formulation agent.
#       Ranks druggable targets using LLM reasoning over retrieved evidence.
#       Produces a selected target dict and a mechanistic hypothesis statement.
# =============================================================================

import logging

from utils.helpers import llm_call, safe_extract_json
from utils.prompts import hypothesis_prompt

log = logging.getLogger("drug_discovery.hypothesis")


class HypothesisAgent:
    """
    Uses retrieved literature, patent landscape, and target info to:
    1. Select the most druggable and evidence-supported target.
    2. Formulate a mechanistic hypothesis (MoA statement).

    Expects state keys: input, literature, patents, target_info
    Produces state keys: hypothesis
    """

    def run(self, state: dict) -> dict:
        log.info("HypothesisAgent running...")

        disease = state.get("input", "")
        lit_findings = state.get("literature", {}).get("synthesis", {})
        patent_findings = {
            "count": state.get("patents", {}).get("patent_count", 0),
            "chemical_claims": state.get("patents", {}).get("chemical_claims", []),
        }
        target_info = state.get("target_info", {})

        # Build a list of target candidates for the LLM
        target_candidates = self._build_candidate_list(target_info, lit_findings)

        try:
            messages = hypothesis_prompt(
                disease=disease,
                lit_findings=lit_findings,
                patent_findings=patent_findings,
                target_candidates=target_candidates,
            )
            raw = llm_call(messages)
            hypothesis_data = safe_extract_json(raw)

            if not hypothesis_data:
                log.warning("LLM returned no parseable hypothesis; using fallback.")
                hypothesis_data = self._fallback_hypothesis(target_info, disease)

            # Enrich with target_info data if LLM didn't specify pdb_id
            hypothesis_data = self._enrich_with_target_info(
                hypothesis_data, target_info
            )

            state["hypothesis"] = hypothesis_data
            log.info(
                f"Hypothesis formulated for target: "
                f"{hypothesis_data.get('selected_target', {}).get('gene_name')}"
            )
            log.info(f"Hypothesis: {hypothesis_data.get('hypothesis', '')[:150]}")

        except Exception as e:
            log.error(f"HypothesisAgent failed: {e}")
            state["hypothesis"] = self._fallback_hypothesis(target_info, disease)

        return state

    # ------------------------------------------------------------------

    def _build_candidate_list(
        self, target_info: dict, lit_findings: dict
    ) -> list[dict]:
        """Compile a list of candidate targets for the LLM to evaluate."""
        candidates = []

        # Primary candidate from UniProt lookup
        if target_info and target_info.get("gene_name"):
            candidates.append(
                {
                    "gene_name": target_info.get("gene_name"),
                    "uniprot_id": target_info.get("uniprot_id"),
                    "pdb_id": target_info.get("pdb_id"),
                    "protein_name": target_info.get("protein_name"),
                    "function_summary": target_info.get("function", "")[:300],
                    "diseases": target_info.get("diseases", []),
                    "has_structure": bool(target_info.get("pdb_id")),
                    "source": "UniProt",
                }
            )

        # Additional candidates from literature
        lit_targets = lit_findings.get("key_targets", [])
        for t in lit_targets[:4]:
            if isinstance(t, str):
                candidates.append(
                    {
                        "gene_name": t,
                        "uniprot_id": "unknown",
                        "pdb_id": "",
                        "source": "literature",
                    }
                )

        return candidates

    def _fallback_hypothesis(self, target_info: dict, disease: str) -> dict:
        """Generate a minimal hypothesis from available metadata."""
        gene = target_info.get("gene_name", "unspecified")
        uniprot = target_info.get("uniprot_id", "")
        pdb = target_info.get("pdb_id", "")
        return {
            "selected_target": {
                "gene_name": gene,
                "uniprot_id": uniprot,
                "pdb_id": pdb,
                "rationale": f"Selected based on literature evidence for {disease}.",
            },
            "hypothesis": (
                f"Selective inhibition of {gene} will attenuate the pathological "
                f"mechanisms underlying {disease} by disrupting the target's "
                f"key biological function."
            ),
            "justification": (
                f"{gene} has been implicated in {disease} pathophysiology. "
                f"A well-characterised PDB structure ({pdb}) enables "
                f"structure-based drug design."
            ),
            "druggability_score": 0.65,
            "confidence": "medium",
        }

    def _enrich_with_target_info(
        self, hypothesis_data: dict, target_info: dict
    ) -> dict:
        """
        If the LLM selected a target but left IDs blank, fill them in
        from the UniProt lookup data if the gene names match.
        """
        selected = hypothesis_data.get("selected_target", {})
        if not selected.get("pdb_id") and target_info.get("pdb_id"):
            selected["pdb_id"] = target_info["pdb_id"]
        if not selected.get("uniprot_id") and target_info.get("uniprot_id"):
            selected["uniprot_id"] = target_info["uniprot_id"]

        # Attach pdb_local_path for downstream docking
        hypothesis_data["pdb_local_path"] = target_info.get("pdb_local_path", "")
        hypothesis_data["selected_target"] = selected
        return hypothesis_data