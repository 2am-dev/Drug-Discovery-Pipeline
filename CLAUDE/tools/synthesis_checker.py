# drug_discovery_pipeline/tools/synthesis_checker.py
# =============================================================================
# FILE: tools/synthesis_checker.py
# ROLE: Synthetic feasibility tool.
#       Retrosynthesis LLM call now routed via routed_llm_call("retrosynthesis")
#       → REMOTE gemma4:31b.
# =============================================================================

import logging
from typing import Optional

from rdkit import Chem

from utils.helpers import routed_llm_call, safe_extract_json
from utils.prompts import retrosynthesis_prompt
from tools.molecule_generator import calculate_sa_score, evaluate_molecule

log = logging.getLogger("drug_discovery.synthesis_checker")


def _try_aizynthfinder(smiles: str) -> Optional[dict]:
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder  # type: ignore
        finder = AiZynthFinder()
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
        stats = finder.extract_statistics()
        return {"source": "aizynthfinder", "solved": stats.get("is_solved", False)}
    except ImportError:
        return None
    except Exception as e:
        log.warning(f"AiZynthFinder error: {e}")
        return None


RETRO_RULES = [
    ("Amide bond hydrolysis",   "[C:1](=[O:2])[N:3]",  "R-COOH",  "R-NH2"),
    ("Ester hydrolysis",        "[C:1](=[O:2])[O:3]",  "R-COOH",  "R-OH"),
    ("Reductive amination",     "[C:1][N:2]",          "R-CHO",   "R-NH2"),
    ("Suzuki coupling",         "[c:1][c:2]",          "Ar-B(OH)2","Ar-X"),
    ("N-alkylation",            "[N:1][C:2]",          "R-NH2",   "R-X"),
    ("Buchwald-Hartwig",        "[c:1][N:2]",          "Ar-X",    "R-NH2 (Pd)"),
    ("Mitsunobu",               "[O:1][C:2]",          "R-OH",    "Nu + DIAD"),
]


def rule_based_retrosynthesis(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["Invalid SMILES"]
    found = []
    for name, smarts, f1, f2 in RETRO_RULES:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            found.append(f"  → {name}: [{f1}] + [{f2}]")
    return found or ["No standard disconnections found"]


def assess_synthesis(smiles: str, target: dict, use_llm: bool = True) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "error": "Invalid SMILES"}

    sa = calculate_sa_score(mol)
    sa_cat = (
        "easy" if sa <= 3 else
        "moderate" if sa <= 5 else
        "challenging" if sa <= 7 else
        "very_challenging"
    )

    disconnections = rule_based_retrosynthesis(smiles)
    aizynth        = _try_aizynthfinder(smiles)

    llm_route = {}
    if use_llm:
        try:
            msgs = retrosynthesis_prompt(smiles, sa, target)
            raw  = routed_llm_call("retrosynthesis", msgs)   # ← REMOTE gemma4:31b
            llm_route = safe_extract_json(raw) or {"raw_response": raw}
        except Exception as e:
            log.error(f"Retrosynthesis LLM failed: {e}")
            llm_route = {"error": str(e)}

    return {
        "smiles":                    smiles,
        "sa_score":                  round(sa, 2),
        "sa_category":               sa_cat,
        "properties":                evaluate_molecule(smiles) or {},
        "rule_based_disconnections": disconnections,
        "aizynthfinder":             aizynth,
        "llm_route":                 llm_route,
        "estimated_steps":           llm_route.get("estimated_steps", max(3, int(sa))),
        "feasibility":               llm_route.get("feasibility", sa_cat),
    }


def assess_multiple(smiles_list: list[str], target: dict, use_llm: bool = True) -> list[dict]:
    results = []
    for s in smiles_list:
        log.info(f"Assessing synthesis: {s[:50]}")
        results.append(assess_synthesis(s, target, use_llm))
    results.sort(key=lambda r: r.get("sa_score", 10.0))
    return results