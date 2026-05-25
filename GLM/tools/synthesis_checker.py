# drug_discovery_pipeline/tools/synthesis_checker.py
"""
Synthetic-accessibility scoring and retrosynthetic-route proposal.
Retrosynthesis LLM call → HEAVY (remote).
Handles empty LLM responses gracefully.
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
        ``smiles``, ``sa_score``, ``feasibility``, ``retrosynthetic_route``
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

    # Retrosynthesis → HEAVY (remote server)
    target_desc = target or "the disease target"
    hypothesis_desc = hypothesis or "modulation of the target activity"
    prompt = RETROSYNTHESIS_PROMPT.format(smiles=smiles, target=target_desc, hypothesis=hypothesis_desc)
    route_text = call_llm(
        prompt,
        system=RETROSYNTHESIS_SYSTEM,
        temperature=0.5,
        max_tokens=1024,
        task="retrosynthesis",
    )

    # Fallback if LLM returned empty
    if not route_text or len(route_text) < 30:
        route_text = (
            f"**Step 1:** Disconnection of the most labile bond in {smiles}.\n"
            f"**Step 2:** Identify commercially available building blocks.\n"
            f"**Step 3:** Couple fragments via standard amide/buchwald coupling.\n\n"
            f"Overall feasibility: {feasibility} (auto-generated fallback)."
        )

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