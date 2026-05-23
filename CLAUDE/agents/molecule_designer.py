# drug_discovery_pipeline/agents/molecule_designer.py
# =============================================================================
# FILE: agents/molecule_designer.py
# ROLE: De novo molecule design agent.
#       1. Extracts pharmacophoric features from known binders via LLM.
#       2. Generates SMILES using fragment-based RDKit generator.
#       3. Filters by Lipinski, QED, SA score, PAINS.
#       4. Asks LLM to rank and comment on the shortlist.
# =============================================================================

import logging

from tools.molecule_generator import (
    evaluate_molecule,
    filter_and_rank,
    generate_and_filter_molecules,
)
from utils.helpers import llm_call, safe_extract_json
from utils.prompts import molecule_refinement_prompt, pharmacophore_prompt

log = logging.getLogger("drug_discovery.molecule_designer")


class MoleculeDesignerAgent:
    """
    Designs novel small-molecule candidates targeting the selected protein.

    Expects state keys: hypothesis, literature, patents
    Produces state keys: pharmacophore, candidates
    """

    def run(self, state: dict) -> dict:
        log.info("MoleculeDesignerAgent running...")

        target = state.get("hypothesis", {}).get("selected_target", {})
        lit_synthesis = state.get("literature", {}).get("synthesis", {})
        patent_claims = state.get("patents", {}).get("chemical_claims", [])

        # ── Step 1: Extract pharmacophore features ─────────────────────
        known_binders = (
            lit_synthesis.get("known_compounds", []) + patent_claims
        )
        pocket_desc = self._describe_pocket(target, state)
        pharmacophore = self._extract_pharmacophore(target, known_binders, pocket_desc)
        state["pharmacophore"] = pharmacophore
        log.info(f"Pharmacophore features: {pharmacophore.get('pharmacophore_features', [])}")

        # ── Step 2: Generate SMILES ────────────────────────────────────
        seed_smiles = [
            s for s in (known_binders + pharmacophore.get("core_scaffolds", []))
            if isinstance(s, str) and len(s) > 5
        ][:10]

        result = generate_and_filter_molecules(
            seed_smiles=seed_smiles,
            pharmacophore=pharmacophore,
        )
        state["candidates"] = result

        log.info(
            f"Generated {result['generation_stats']['total_generated']} molecules; "
            f"{result['generation_stats']['drug_like']} drug-like; "
            f"shortlisted {result['generation_stats']['shortlisted']}."
        )

        # ── Step 3: LLM refinement commentary ─────────────────────────
        if result["shortlist"]:
            shortlist_smiles = [m["smiles"] for m in result["shortlist"]]
            state["candidates"]["llm_commentary"] = self._llm_refinement(
                shortlist_smiles, target, pharmacophore
            )

        return state

    # ------------------------------------------------------------------

    def _describe_pocket(self, target: dict, state: dict) -> str:
        """
        Build a text description of the binding pocket from available data.
        """
        gene = target.get("gene_name", "Unknown")
        pdb = target.get("pdb_id", "N/A")
        hypothesis_text = state.get("hypothesis", {}).get("hypothesis", "")
        return (
            f"Target: {gene} (PDB: {pdb}). "
            f"Mechanistic context: {hypothesis_text[:300]}"
        )

    def _extract_pharmacophore(
        self,
        target: dict,
        known_binders: list[str],
        pocket_desc: str,
    ) -> dict:
        """Call LLM to extract pharmacophoric features."""
        try:
            messages = pharmacophore_prompt(target, known_binders, pocket_desc)
            raw = llm_call(messages)
            data = safe_extract_json(raw)
            if data:
                return data
        except Exception as e:
            log.error(f"Pharmacophore extraction failed: {e}")

        # Fallback pharmacophore
        return {
            "pharmacophore_features": [
                "H-bond donor (NH or OH)",
                "H-bond acceptor (C=O or N)",
                "Aromatic ring",
                "Hydrophobic region",
            ],
            "core_scaffolds": [
                "c1ccc2[nH]cccc2c1",  # indole
                "c1ccc2ncccc2c1",     # quinoline
            ],
            "forbidden_groups": [
                "aldehyde (CHO)",
                "Michael acceptor",
                "catechol",
            ],
            "design_strategy": "Fragment growing from aromatic core",
        }

    def _llm_refinement(
        self,
        smiles_list: list[str],
        target: dict,
        pharmacophore: dict,
    ) -> dict:
        """Ask LLM to comment on and rank the shortlisted candidates."""
        try:
            messages = molecule_refinement_prompt(smiles_list, target, pharmacophore)
            raw = llm_call(messages)
            return safe_extract_json(raw) or {"raw": raw}
        except Exception as e:
            log.error(f"LLM refinement failed: {e}")
            return {"error": str(e)}