# drug_discovery_pipeline/agents/molecule_designer.py
"""
Molecule Designer Agent.
Pharmacophore extraction → LIGHT (local).
Molecule generation → pure RDKit (no LLM).
Handles empty hypothesis gracefully.
"""

from __future__ import annotations

import logging
from typing import Dict

from utils.helpers import call_llm
from utils.prompts import PHARMACOPHORE_PROMPT
from tools.molecule_generator import generate_molecules, filter_molecules
from config import NUM_CANDIDATES_GENERATE, NUM_CANDIDATES_SHORTLIST

logger = logging.getLogger(__name__)


class MoleculeDesignerAgent:
    """Designs novel small-molecule drug candidates in silico."""

    def run(self, state: Dict) -> Dict:
        logger.info("MoleculeDesignerAgent starting…")

        hypothesis = state.get("hypothesis", {})
        target = hypothesis.get("target", "") or state.get("input", "")
        uniprot_id = hypothesis.get("uniprot_id", "") or "unknown"
        hypothesis_text = hypothesis.get("hypothesis", "") or (
            f"A small-molecule modulator of {target} will reduce disease pathology."
        )

        # ── Pharmacophore → LIGHT (local server) ─────────────────────────
        pharma_prompt = PHARMACOPHORE_PROMPT.format(
            target=target,
            uniprot_id=uniprot_id,
            hypothesis=hypothesis_text,
        )
        pharmacophore = call_llm(
            pharma_prompt,
            temperature=0.4,
            max_tokens=512,
            task="pharmacophore",
        )

        # Fallback if pharmacophore is empty or unhelpful
        if not pharmacophore or "haven't provided" in pharmacophore.lower() or len(pharmacophore) < 30:
            pharmacophore = (
                "- Hydrogen-bond acceptor (N or O) in the binding pocket\n"
                "- Hydrophobic aromatic ring for π-stacking\n"
                "- Hydrogen-bond donor for key polar interaction\n"
                "- Moderate molecular weight (200-400 Da) region\n"
                "- Optional halogen substituent for enhanced binding"
            )
            logger.warning("Using default pharmacophore (LLM response was empty/unhelpful).")

        state["pharmacophore"] = pharmacophore
        logger.info("Pharmacophore features:\n%s", pharmacophore[:300])

        # ── Generate molecules (RDKit only – no LLM) ────────────────────
        raw_molecules = generate_molecules(
            pharmacophore_hints=pharmacophore,
            n=NUM_CANDIDATES_GENERATE,
        )
        state["generated_molecules"] = [
            {"smiles": m["smiles"], "qed": m["qed"], "sa_score": m["sa_score"]}
            for m in raw_molecules
        ]
        logger.info("Generated %d raw molecules.", len(raw_molecules))

        # ── Filter and shortlist ─────────────────────────────────────────
        top_molecules = filter_molecules(raw_molecules, top_k=NUM_CANDIDATES_SHORTLIST)
        state["shortlisted_molecules"] = [
            {
                "smiles": m["smiles"],
                "qed": m["qed"],
                "sa_score": m["sa_score"],
                "composite_score": m.get("composite_score", 0),
            }
            for m in top_molecules
        ]
        logger.info("Shortlisted %d molecules.", len(top_molecules))

        return state