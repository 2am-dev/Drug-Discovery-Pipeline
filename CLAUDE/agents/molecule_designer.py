# drug_discovery_pipeline/agents/molecule_designer.py
# =============================================================================
# FILE: agents/molecule_designer.py
# ROLE: De novo molecule design agent.
#       Pharmacophore extraction   → REMOTE gemma4:31b  (task="pharmacophore")
#       Molecule refinement/rank   → REMOTE gemma4:31b  (task="molecule_refine")
#       SMILES generation/filtering → RDKit (no LLM needed)
# =============================================================================

import logging

from tools.molecule_generator import generate_and_filter_molecules
from utils.helpers import routed_llm_call, safe_extract_json
from utils.prompts import molecule_refinement_prompt, pharmacophore_prompt

log = logging.getLogger("drug_discovery.molecule_designer")


class MoleculeDesignerAgent:
    """
    Pharmacophore extraction + molecule ranking → REMOTE gemma4:31b
    SMILES generation + filtering              → RDKit (local CPU/GPU)
    """

    def run(self, state: dict) -> dict:
        log.info("MoleculeDesignerAgent running...")
        target       = state.get("hypothesis", {}).get("selected_target", {})
        lit_synthesis = state.get("literature", {}).get("synthesis", {})
        patent_claims = state.get("patents", {}).get("chemical_claims", [])

        known_binders = lit_synthesis.get("known_compounds", []) + patent_claims
        pocket_desc   = self._describe_pocket(target, state)

        # ── Pharmacophore (REMOTE) ─────────────────────────────────────
        pharmacophore = self._extract_pharmacophore(target, known_binders, pocket_desc)
        state["pharmacophore"] = pharmacophore

        # ── SMILES generation (RDKit, no LLM) ─────────────────────────
        seed = [s for s in (known_binders + pharmacophore.get("core_scaffolds", []))
                if isinstance(s, str) and len(s) > 5][:10]
        result = generate_and_filter_molecules(seed_smiles=seed, pharmacophore=pharmacophore)
        state["candidates"] = result

        log.info(
            f"Generated {result['generation_stats']['total_generated']} | "
            f"drug-like {result['generation_stats']['drug_like']} | "
            f"shortlisted {result['generation_stats']['shortlisted']}"
        )

        # ── LLM refinement commentary (REMOTE) ────────────────────────
        if result["shortlist"]:
            smi_list = [m["smiles"] for m in result["shortlist"]]
            state["candidates"]["llm_commentary"] = self._llm_refinement(
                smi_list, target, pharmacophore
            )
        return state

    def _describe_pocket(self, target: dict, state: dict) -> str:
        hyp_text = state.get("hypothesis", {}).get("hypothesis", "")
        return (
            f"Target: {target.get('gene_name','Unknown')} "
            f"(PDB: {target.get('pdb_id','N/A')}). "
            f"Context: {hyp_text[:300]}"
        )

    def _extract_pharmacophore(self, target: dict, binders: list, pocket: str) -> dict:
        try:
            messages = pharmacophore_prompt(target, binders, pocket)
            raw  = routed_llm_call("pharmacophore", messages)   # ← REMOTE
            data = safe_extract_json(raw)
            if data:
                return data
        except Exception as e:
            log.error(f"Pharmacophore extraction failed: {e}")
        return {
            "pharmacophore_features": [
                "H-bond donor (NH/OH)", "H-bond acceptor (C=O/N)",
                "Aromatic ring", "Hydrophobic region",
            ],
            "core_scaffolds":   ["c1ccc2[nH]cccc2c1", "c1ccc2ncccc2c1"],
            "forbidden_groups": ["aldehyde (CHO)", "Michael acceptor", "catechol"],
            "design_strategy":  "Fragment growing from aromatic core",
        }

    def _llm_refinement(self, smiles: list, target: dict, pharma: dict) -> dict:
        try:
            messages = molecule_refinement_prompt(smiles, target, pharma)
            raw = routed_llm_call("molecule_refine", messages)   # ← REMOTE
            return safe_extract_json(raw) or {"raw": raw}
        except Exception as e:
            log.error(f"Molecule refinement failed: {e}")
            return {"error": str(e)}