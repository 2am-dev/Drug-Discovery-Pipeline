# drug_discovery_pipeline/tools/synthesis_checker.py
"""
Synthetic-accessibility scoring and retrosynthetic-route proposal.
SA Score is computed locally via RDKit; retrosynthesis is proposed by the LLM.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from utils.helpers import call_llm
from utils.prompts import RETROSYNTHESIS_SYSTEM, RETROSYNTHESIS_PROMPT
from tools.molecule_generator import compute_sa_score

from rdkit import Chem

logger = logging.getLogger(__name__)


def evaluate_synthesis(smiles: str, target: str, hypothesis: str) -> Dict:
    """
    Evaluate synthetic accessibility for a single molecule.

    Returns a dict with keys:
        ``smiles``, ``sa_score``, ``feasibility`` (Easy / Medium / Hard),
        ``retrosynthetic_route`` (text from LLM)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "sa_score": 10.0, "feasibility": "Invalid", "retrosynthetic_route": ""}

    sa = compute_sa_score(mol)

    if sa <= 3.5:
        feasibility = "Easy"
    elif sa <= 6.0:
        feasibility = "Medium"
    else:
        feasibility = "Hard"

    # Use LLM to propose a retrosynthetic route
    prompt = RETROSYNTHESIS_PROMPT.format(smiles=smiles, target=target, hypothesis=hypothesis)
    route_text = call_llm(prompt, system=RETROSYNTHESIS_SYSTEM, temperature=0.5, max_tokens=1024)

    return {
        "smiles": smiles,
        "sa_score": sa,
        "feasibility": feasibility,
        "retrosynthetic_route": route_text,
    }


def batch_evaluate(smiles_list: List[str], target: str, hypothesis: str) -> List[Dict]:
    """Run :func:`evaluate_synthesis` on a batch of molecules."""
    results: list[dict] = []
    for smi in smiles_list:
        logger.info("Evaluating synthesis for: %s", smi)
        result = evaluate_synthesis(smi, target, hypothesis)
        results.append(result)
    return results