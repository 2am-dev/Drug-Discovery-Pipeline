# drug_discovery_pipeline/tools/molecule_generator.py
# =============================================================================
# FILE: tools/molecule_generator.py
# ROLE: De novo small-molecule generation tool.
#       Uses a fragment-based, scaffold-decoration approach with RDKit.
#       Filters candidates by Lipinski rules, QED, SA score, and PAINS.
#       Returns a shortlist of drug-like SMILES.
# =============================================================================

import logging
import random
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, QED, RWMol
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from config import (
    LIPINSKI_HBA_MAX,
    LIPINSKI_HBD_MAX,
    LIPINSKI_LOGP_MAX,
    LIPINSKI_MW_MAX,
    NUM_CANDIDATES_GENERATE,
    NUM_CANDIDATES_SHORTLIST,
    QED_MIN,
    SA_SCORE_MAX,
)

log = logging.getLogger("drug_discovery.molecule_generator")

# ---------------------------------------------------------------------------
# SA Score (RDKit contrib – loaded lazily)
# ---------------------------------------------------------------------------

_sa_score_fn = None


def _get_sa_score_fn():
    """Lazily load the SA score function from RDKit contrib."""
    global _sa_score_fn
    if _sa_score_fn is not None:
        return _sa_score_fn
    try:
        from rdkit.Chem.rdMolDescriptors import CalcCrippenDescriptors  # noqa
        # Try the standard RDKit sascorer
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        _sa_score_fn = sascorer.calculateScore
        log.info("SA score loaded from rdkit.Contrib.SA_Score")
    except ImportError:
        try:
            import sys, importlib.util, os
            # Try to find sa_score in common locations
            for candidate in [
                "sascorer",
                "rdkit.Contrib.SA_Score.sascorer",
            ]:
                try:
                    mod = __import__(candidate, fromlist=["calculateScore"])
                    _sa_score_fn = mod.calculateScore
                    break
                except Exception:
                    pass
        except Exception:
            pass

    if _sa_score_fn is None:
        log.warning(
            "SA score function not found; using MW-based heuristic fallback."
        )
        _sa_score_fn = _sa_score_heuristic
    return _sa_score_fn


def _sa_score_heuristic(mol) -> float:
    """
    Rough heuristic SA score when the real sascorer is unavailable.
    Based on ring complexity and number of stereocentres.
    Range: 1 (easy) – 10 (hard).
    """
    try:
        ring_info = mol.GetRingInfo()
        n_rings = ring_info.NumRings()
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        mw = Descriptors.MolWt(mol)
        score = 1.0 + (n_rings * 0.4) + (n_stereo * 0.6) + (mw / 500.0)
        return min(score, 10.0)
    except Exception:
        return 5.0


def calculate_sa_score(mol) -> float:
    """Calculate the synthetic accessibility score for *mol*."""
    fn = _get_sa_score_fn()
    try:
        return float(fn(mol))
    except Exception as e:
        log.warning(f"SA score calculation error: {e}")
        return 5.0


# ---------------------------------------------------------------------------
# PAINS filter
# ---------------------------------------------------------------------------

def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


_PAINS_CATALOG = None


def is_pains(mol) -> bool:
    """Return True if *mol* matches any PAINS substructure pattern."""
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        _PAINS_CATALOG = _build_pains_catalog()
    return _PAINS_CATALOG.HasMatch(mol)


# ---------------------------------------------------------------------------
# Drug-likeness filters
# ---------------------------------------------------------------------------

def lipinski_pass(mol) -> tuple[bool, dict]:
    """
    Evaluate Lipinski Ro5. Returns (pass_bool, property_dict).
    Allows up to 1 violation (Pfizer rule).
    """
    props = {
        "MW": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": CalcTPSA(mol),
    }
    violations = 0
    if props["MW"] > LIPINSKI_MW_MAX:
        violations += 1
    if props["LogP"] > LIPINSKI_LOGP_MAX:
        violations += 1
    if props["HBD"] > LIPINSKI_HBD_MAX:
        violations += 1
    if props["HBA"] > LIPINSKI_HBA_MAX:
        violations += 1
    return violations <= 1, props


def calculate_qed(mol) -> float:
    """Return the QED drug-likeness score (0-1)."""
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Fragment library
# ---------------------------------------------------------------------------

# Curated set of drug-like scaffolds and fragments
# These are real scaffolds found in approved drugs / clinical candidates
DRUG_SCAFFOLDS = [
    # Heterocyclic cores
    "c1ccccc1",              # benzene
    "c1ccncc1",              # pyridine
    "c1ccoc1",               # furan
    "c1ccsc1",               # thiophene
    "c1cncc1",               # imidazole scaffold
    "c1ccc2ncccc2c1",        # quinoline
    "c1ccc2ccccc2n1",        # isoquinoline
    "c1ccc2[nH]cccc2c1",     # indole
    "c1cnc2ccccc2n1",        # benzimidazole
    "c1ccc2occc2c1",         # benzofuran
    "c1ccc2sccc2c1",         # benzothiophene
    "C1CCNCC1",              # piperidine
    "C1COCCN1",              # morpholine
    "C1CNCCN1",              # piperazine
    "C1CCC(=O)N1",           # pyrrolidinone
    "c1ccc(cc1)C(=O)N",      # benzamide
    "c1ccc(cc1)S(=O)(=O)N",  # benzenesulfonamide
    "c1ccc(cc1)OCC",         # phenoxyethane
    "c1ccc(cc1)NC(=O)",      # aniline amide
]

LINKERS = [
    "CC", "CCC", "CCCC",
    "C(=O)", "C(=O)O", "C(=O)N",
    "CC(=O)", "CNC(=O)",
    "CS", "CO", "CN",
    "c1ccc(cc1)",  # para-phenylene linker
    "c1cc(cc(c1))",  # meta-phenylene
]

TERMINAL_GROUPS = [
    "N", "O", "F", "Cl", "Br",
    "C(=O)O", "C(=O)N",
    "S(=O)(=O)N", "S(=O)(=O)O",
    "C#N", "OC",
    "NC(=O)",
    "c1ccccc1",
    "C(F)(F)F",
    "OCC",
]


# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------

def _smiles_to_mol(smiles: str) -> Optional[object]:
    """Safe SMILES → RDKit Mol conversion."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
    except Exception:
        pass
    return None


def generate_from_scaffold(
    scaffold_smiles: str,
    n_variants: int = 5,
    rng: random.Random = None,
) -> list[str]:
    """
    Generate *n_variants* SMILES variants by randomly combining a scaffold
    with linkers and terminal groups.  Uses simple SMILES string manipulation
    (not proper combinatorial chemistry – this is a demonstration scaffold).
    """
    if rng is None:
        rng = random.Random()

    candidates = []
    scaffold = _smiles_to_mol(scaffold_smiles)
    if scaffold is None:
        return candidates

    for _ in range(n_variants * 3):  # oversample and filter
        try:
            linker = rng.choice(LINKERS)
            terminal = rng.choice(TERMINAL_GROUPS)
            # Naive concatenation – works for open-valence atoms
            candidate_smiles = f"{scaffold_smiles}{linker}{terminal}"
            mol = _smiles_to_mol(candidate_smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                if canonical not in candidates:
                    candidates.append(canonical)
                if len(candidates) >= n_variants:
                    break
        except Exception:
            continue
    return candidates


def generate_smiles_library(
    seed_smiles: list[str] = None,
    n_total: int = NUM_CANDIDATES_GENERATE,
    rng_seed: int = 42,
) -> list[str]:
    """
    Generate a diverse SMILES library.

    Args:
        seed_smiles: Known binder SMILES to use as primary scaffolds.
        n_total: Target number of candidates to generate.
        rng_seed: Random seed for reproducibility.

    Returns:
        List of canonical SMILES strings.
    """
    rng = random.Random(rng_seed)
    all_candidates: list[str] = []

    # Use seed SMILES as primary scaffolds
    primary_scaffolds = []
    if seed_smiles:
        for s in seed_smiles[:5]:
            mol = _smiles_to_mol(s)
            if mol is not None:
                primary_scaffolds.append(Chem.MolToSmiles(mol))

    # Supplement with built-in drug scaffolds
    secondary_scaffolds = rng.sample(DRUG_SCAFFOLDS, min(8, len(DRUG_SCAFFOLDS)))
    scaffolds = (primary_scaffolds + secondary_scaffolds)[:12]

    per_scaffold = max(2, n_total // len(scaffolds))
    for scaffold in scaffolds:
        variants = generate_from_scaffold(scaffold, n_variants=per_scaffold, rng=rng)
        all_candidates.extend(variants)

    # Deduplicate
    seen = set()
    unique = []
    for s in all_candidates:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    rng.shuffle(unique)
    return unique[:n_total]


# ---------------------------------------------------------------------------
# Filtering pipeline
# ---------------------------------------------------------------------------

def evaluate_molecule(smiles: str) -> Optional[dict]:
    """
    Evaluate a single SMILES against all drug-likeness criteria.
    Returns a property dict, or None if the molecule is invalid.
    """
    mol = _smiles_to_mol(smiles)
    if mol is None:
        return None

    lipinski_ok, props = lipinski_pass(mol)
    qed_score = calculate_qed(mol)
    sa = calculate_sa_score(mol)
    pains = is_pains(mol)

    return {
        "smiles": Chem.MolToSmiles(mol),
        "MW": round(props["MW"], 2),
        "LogP": round(props["LogP"], 2),
        "HBD": props["HBD"],
        "HBA": props["HBA"],
        "TPSA": round(props["TPSA"], 2),
        "QED": round(qed_score, 3),
        "SA_Score": round(sa, 2),
        "Lipinski_Pass": lipinski_ok,
        "PAINS": pains,
        "drug_like": lipinski_ok and qed_score >= QED_MIN and sa <= SA_SCORE_MAX and not pains,
    }


def filter_and_rank(
    smiles_list: list[str],
    top_k: int = NUM_CANDIDATES_SHORTLIST,
) -> list[dict]:
    """
    Evaluate and rank a list of SMILES.
    Filters by drug-likeness; ranks by a composite score (QED ↑, SA ↓).

    Returns the top *top_k* evaluated molecule dicts.
    """
    evaluated = []
    for smiles in smiles_list:
        result = evaluate_molecule(smiles)
        if result is not None:
            evaluated.append(result)

    # Keep only drug-like molecules
    drug_like = [m for m in evaluated if m["drug_like"]]
    log.info(f"Drug-like molecules: {len(drug_like)}/{len(evaluated)}")

    # If too few pass strict filter, relax
    if len(drug_like) < top_k:
        drug_like = sorted(
            evaluated,
            key=lambda m: (not m["Lipinski_Pass"], -m["QED"], m["SA_Score"]),
        )

    # Composite score: maximise QED, minimise SA_Score (normalised)
    for m in drug_like:
        m["composite_score"] = m["QED"] - (m["SA_Score"] / 20.0)

    drug_like.sort(key=lambda m: m["composite_score"], reverse=True)
    return drug_like[:top_k]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_and_filter_molecules(
    seed_smiles: list[str] = None,
    pharmacophore: dict = None,
    n_generate: int = NUM_CANDIDATES_GENERATE,
    n_shortlist: int = NUM_CANDIDATES_SHORTLIST,
) -> dict:
    """
    End-to-end molecule generation and filtering.

    Returns:
        {
            "all_candidates": list of evaluated dicts,
            "shortlist": top-k drug-like candidates,
            "generation_stats": {total, valid, drug_like, shortlisted},
        }
    """
    log.info(f"Generating {n_generate} SMILES candidates...")

    # Extract seed SMILES from pharmacophore if available
    if pharmacophore and not seed_smiles:
        seed_smiles = pharmacophore.get("core_scaffolds", [])

    raw_smiles = generate_smiles_library(seed_smiles=seed_smiles, n_total=n_generate)
    log.info(f"Generated {len(raw_smiles)} raw SMILES.")

    shortlist = filter_and_rank(raw_smiles, top_k=n_shortlist)
    all_evaluated = [evaluate_molecule(s) for s in raw_smiles if evaluate_molecule(s)]

    return {
        "all_candidates": all_evaluated,
        "shortlist": shortlist,
        "generation_stats": {
            "total_generated": len(raw_smiles),
            "valid_molecules": len(all_evaluated),
            "drug_like": len([m for m in all_evaluated if m["drug_like"]]),
            "shortlisted": len(shortlist),
        },
    }