# drug_discovery_pipeline/agents/molecule_designer.py
"""
Molecule Designer Agent.

1. Extracts pharmacophore features via LLM.
2. Generates candidate molecules using the fragment/mutation engine.
3. Filters and shortlists the best candidates.
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
        target = hypothesis.get("target", state.get("input", ""))
        uniprot_id = hypothesis.get("uniprot_id", "")
        hypothesis_text = hypothesis.get("hypothesis", "")

        # ── Extract pharmacophore features ───────────────────────────────
        pharma_prompt = PHARMACOPHORE_PROMPT.format(
            target=target,
            uniprot_id=uniprot_id,
            hypothesis=hypothesis_text,
        )
        pharmacophore = call_llm(pharma_prompt, temperature=0.4, max_tokens=512)
        state["pharmacophore"] = pharmacophore
        logger.info("Pharmacophore features extracted:\n%s", pharmacophore[:300])

        # ── Generate molecules ───────────────────────────────────────────
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