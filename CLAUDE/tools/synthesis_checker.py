# drug_discovery_pipeline/tools/synthesis_checker.py
# =============================================================================
# FILE: tools/synthesis_checker.py
# ROLE: Synthetic feasibility assessment tool.
#       Computes SA Score and Lipinski properties for each candidate.
#       Calls the LLM to propose a retrosynthetic route.
#       Optionally interfaces with AiZynthFinder (if installed).
# =============================================================================

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

from utils.helpers import llm_call, safe_extract_json
from utils.prompts import retrosynthesis_prompt
from tools.molecule_generator import calculate_sa_score, evaluate_molecule

log = logging.getLogger("drug_discovery.synthesis_checker")


# ---------------------------------------------------------------------------
# AiZynthFinder integration (optional)
# ---------------------------------------------------------------------------

def _try_aizynthfinder(smiles: str) -> Optional[dict]:
    """
    Attempt to use AiZynthFinder for retrosynthesis if installed.
    Returns a route dict or None.
    """
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder  # type: ignore
        log.info("AiZynthFinder found; running retrosynthesis...")
        # Minimal config – user should provide a proper config file
        finder = AiZynthFinder()
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
        stats = finder.extract_statistics()
        return {
            "source": "aizynthfinder",
            "solved": stats.get("is_solved", False),
            "routes": str(stats),
        }
    except ImportError:
        return None
    except Exception as e:
        log.warning(f"AiZynthFinder failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Reaction template library (rule-based fallback)
# ---------------------------------------------------------------------------

# Simple SMARTS-based disconnection rules
# Format: (name, product_smarts, fragment_1_hint, fragment_2_hint)
RETRO_RULES = [
    (
        "Amide bond hydrolysis",
        "[C:1](=[O:2])[N:3]",
        "carboxylic acid R-COOH",
        "amine R-NH2",
    ),
    (
        "Ester hydrolysis",
        "[C:1](=[O:2])[O:3]",
        "carboxylic acid R-COOH",
        "alcohol R-OH",
    ),
    (
        "Reductive amination",
        "[C:1][N:2]",
        "aldehyde or ketone R-CHO/R-CO-R",
        "amine R-NH2",
    ),
    (
        "Suzuki coupling",
        "[c:1][c:2]",
        "aryl boronic acid Ar-B(OH)2",
        "aryl halide Ar-X",
    ),
    (
        "N-alkylation",
        "[N:1][C:2]",
        "amine R-NH2",
        "alkyl halide R-X",
    ),
    (
        "Buchwald-Hartwig",
        "[c:1][N:2]",
        "aryl halide Ar-X",
        "amine R-NH2 (Pd-catalysed)",
    ),
    (
        "Mitsunobu reaction",
        "[O:1][C:2]",
        "alcohol R-OH",
        "nucleophile + DIAD/PPh3",
    ),
]


def rule_based_retrosynthesis(smiles: str) -> list[str]:
    """
    Apply SMARTS-based disconnection rules to identify possible retrosynthetic
    steps. Returns a list of descriptive strings.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["Invalid SMILES – cannot perform retrosynthesis"]

    disconnections = []
    for name, smarts, frag1, frag2 in RETRO_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            disconnections.append(
                f"  → {name}: disconnect to [{frag1}] + [{frag2}]"
            )

    if not disconnections:
        disconnections = ["No standard disconnections identified; custom route required"]

    return disconnections


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def assess_synthesis(
    smiles: str,
    target: dict,
    use_llm: bool = True,
) -> dict:
    """
    Comprehensive synthetic feasibility assessment for a single SMILES.

    Returns a dict:
    {
        smiles, sa_score, sa_category,
        rule_based_disconnections,
        llm_route,          # from LLM retrosynthesis prompt
        aizynthfinder,      # from AiZynthFinder (if available)
        estimated_steps,
        feasibility,
    }
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "error": "Invalid SMILES"}

    sa = calculate_sa_score(mol)
    props = evaluate_molecule(smiles) or {}

    if sa <= 3.0:
        sa_cat = "easy"
    elif sa <= 5.0:
        sa_cat = "moderate"
    elif sa <= 7.0:
        sa_cat = "challenging"
    else:
        sa_cat = "very_challenging"

    # Rule-based disconnections
    disconnections = rule_based_retrosynthesis(smiles)

    # AiZynthFinder (optional)
    aizynth_result = _try_aizynthfinder(smiles)

    # LLM retrosynthesis
    llm_route = {}
    if use_llm:
        try:
            messages = retrosynthesis_prompt(smiles, sa, target)
            raw = llm_call(messages)
            llm_route = safe_extract_json(raw) or {"raw_response": raw}
        except Exception as e:
            log.error(f"LLM retrosynthesis failed: {e}")
            llm_route = {"error": str(e)}

    # Estimate steps from LLM output or fallback
    estimated_steps = llm_route.get("estimated_steps", max(3, int(sa)))
    feasibility = llm_route.get("feasibility", sa_cat)

    return {
        "smiles": smiles,
        "sa_score": round(sa, 2),
        "sa_category": sa_cat,
        "properties": props,
        "rule_based_disconnections": disconnections,
        "aizynthfinder": aizynth_result,
        "llm_route": llm_route,
        "estimated_steps": estimated_steps,
        "feasibility": feasibility,
    }


def assess_multiple(
    smiles_list: list[str],
    target: dict,
    use_llm: bool = True,
) -> list[dict]:
    """
    Assess synthetic feasibility for a list of SMILES.
    Returns a list sorted by SA score (easiest first).
    """
    results = []
    for smiles in smiles_list:
        log.info(f"Assessing synthesis for: {smiles[:50]}")
        result = assess_synthesis(smiles, target, use_llm=use_llm)
        results.append(result)

    results.sort(key=lambda r: r.get("sa_score", 10.0))
    return results