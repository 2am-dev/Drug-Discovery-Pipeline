# drug_discovery_pipeline/tools/molecule_generator.py
"""
Molecule generation using scaffold-based fragment combination and
rule-based mutation.  All generation is done locally with RDKit.

Fixes:
  - Corrected R-group SMILES (CF3 → C(F)(F)F, etc.)
  - Added minimum heavy-atom count filter (≥ 8)
  - Improved fragment combination robustness
"""

from __future__ import annotations

import logging
import random
import sys
import os
from typing import List, Dict, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.QED import qed
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

logger = logging.getLogger(__name__)

# ── SA-Score import ───────────────────────────────────────────────────────────

HAS_SA_SCORE = False
try:
    from rdkit.Chem import RDConfig
    _sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if os.path.isdir(_sa_path):
        sys.path.insert(0, _sa_path)
        import sascorer  # type: ignore
        HAS_SA_SCORE = True
except Exception:
    pass


# ── PAINS filter catalog ─────────────────────────────────────────────────────

def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


_PAINS_CATALOG = _build_pains_catalog()

# ── Fragment libraries (FIXED SMILES) ─────────────────────────────────────────

MONO_SCAFFOLDS: list[str] = [
    "[*]c1ccccc1",           # phenyl
    "[*]c1cccnc1",           # pyridine-2
    "[*]c1ccncc1",           # pyridine-3
    "[*]c1cncnc1",           # pyrimidine
    "[*]c1ccc2[nH]ccc2c1",   # indole
    "[*]c1ccc2ccccc2c1",     # naphthalene
    "[*]C(=O)N1CCCC1",       # pyrrolidinone
    "[*]c1ccc2c(c1)nc(nc2)=O", # quinazolinone
    "[*]c1ccco1",            # furan
    "[*]c1cccs1",            # thiophene
    "[*]c1ccn[nH]1",         # pyrazole
    "[*]C1CCNCC1",           # piperidine
    "[*]c1ccc2c(c1)CCNC2=O", # isoindolinone
    "[*]c1nc2ccccc2n1C",     # benzimidazole
    "[*]c1ccc2c(c1)CCNC2",   # isoindoline
    "[*]C2=C1C=CC=CC1=N2",   # quinoline
]

DI_SCAFFOLDS: list[str] = [
    "[*]c1ccc(cc1)[*]",            # para-phenyl
    "[*]c1ccc(nc1)[*]",            # para-pyridine
    "[*]c1cnc(nc1)[*]",            # pyrimidine-4,6
    "[*]N1CCN(CC1)[*]",            # piperazine
    "[*]C(=O)NC([*])=O",           # oxalamide
    "[*]c1ccc2c(c1)ccc(nc2)[*]",   # quinoline-6,7
    "[*]c1ccc(cc1)C(=O)[*]",       # benzoyl
    "[*]c1ccc(cc1)O[*]",           # diphenyl ether
]

R_GROUPS: list[str] = [
    "[*]C",
    "[*]CC",
    "[*]CCC",
    "[*]C(C)C",         # isopropyl
    "[*]O",
    "[*]OC",
    "[*]OCC",
    "[*]F",
    "[*]Cl",
    "[*]C(F)(F)F",      # trifluoromethyl  ← FIXED
    "[*]NC",
    "[*]NCC",
    "[*]C(=O)C",
    "[*]C(=O)OC",
    "[*]C(=O)N(C)C",
    "[*]SC",
    "[*]S(=O)(=O)C",    # methylsulfonyl   ← FIXED
    "[*]CN",
    "[*]C(=O)NC",
    "[*]N1CCCC1",        # pyrrolidine
    "[*]OCCF",           # 2-fluoroethoxy
    "[*]C#N",            # cyano
    "[*]OC(F)(F)F",      # trifluoromethoxy ← FIXED
]

SEED_MOLECULES: list[str] = [
    "CC(=O)Oc1ccccc1C(=O)O",                          # aspirin scaffold
    "c1ccccc1C(=O)NCC2CCNCC2",                        # benzamide-piperidine
    "Cc1nc2ccccc2c(=O)n1C",                            # phenytoin-like
    "CC(=O)Nc1cccc(O)c1",                              # acetaminophen-like
    "Cc1ccc2c(c1)cc(nc2=O)N3CCCC3",                   # quinazolinone-pyrrolidine
    "Cc1ccnc(N2CCN(C)CC2)c1",                          # aminopyridine-piperazine
    "c1ccccc1S(=O)(=O)NC2CCNCC2",                     # sulfonylpiperazine
    "O=C1NC2=CC=CC=C2C(=O)N1C",                       # phthalimide
    "FCc1ccc(O)c(O)c1",                                # fluorocatechol
    "c1ccc(-c2cccnc2)cc1",                             # biphenyl-pyridine
    "CC(=O)Nc1ccc(O)cc1",                              # acetaminophen
    "O=c1nc(-c2ccccc2)nc2ccccc12",                    # quinazoline-phenyl
    "c1ccccc1C(=O)c2ccccc2",                           # benzophenone
    "CC(C)c1ccc(O)cc1",                                # butylphenol
    "c1ccnc(Nc2ccccc2)c1",                             # aminopyridine-phenyl
]


# ── Fragment combination ─────────────────────────────────────────────────────

def _combine_at_dummy(scaffold_smiles: str, rgroup_smiles_list: list[str]) -> Optional[Chem.Mol]:
    """
    Combine a scaffold with R-groups at ``[*]`` attachment points.
    Processes one R-group at a time.
    """
    current = Chem.MolFromSmiles(scaffold_smiles)
    if current is None:
        return None

    for rg_smiles in rgroup_smiles_list:
        rg = Chem.MolFromSmiles(rg_smiles)
        if rg is None:
            return None

        # Find a dummy atom in current molecule
        scaff_dummy_idx = None
        scaff_nbr_idx = None
        for atom in current.GetAtoms():
            if atom.GetAtomicNum() == 0:
                nbrs = [n for n in atom.GetNeighbors()]
                if nbrs:
                    scaff_dummy_idx = atom.GetIdx()
                    scaff_nbr_idx = nbrs[0].GetIdx()
                    break
        if scaff_dummy_idx is None:
            return None

        # Find dummy in R-group
        rg_dummy_idx = None
        rg_nbr_idx = None
        for atom in rg.GetAtoms():
            if atom.GetAtomicNum() == 0:
                nbrs = [n for n in atom.GetNeighbors()]
                if nbrs:
                    rg_dummy_idx = atom.GetIdx()
                    rg_nbr_idx = nbrs[0].GetIdx()
                    break
        if rg_dummy_idx is None or rg_nbr_idx is None:
            return None

        # Combine and modify
        combined = Chem.CombineMols(current, rg)
        rw = Chem.RWMol(combined)
        n_current = current.GetNumAtoms()
        rg_nbr_combined = rg_nbr_idx + n_current
        rg_dummy_combined = rg_dummy_idx + n_current

        rw.AddBond(scaff_nbr_idx, rg_nbr_combined, Chem.BondType.SINGLE)

        for idx in sorted([scaff_dummy_idx, rg_dummy_combined], reverse=True):
            rw.RemoveAtom(idx)

        try:
            Chem.SanitizeMol(rw)
            current = Chem.Mol(rw)
        except Exception:
            return None

    return current


# ── Mutation functions ────────────────────────────────────────────────────────

def _add_fluorine(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    candidates = [
        a.GetIdx() for a in rw.GetAtoms()
        if a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
    ]
    if not candidates:
        return None
    idx = random.choice(candidates)
    f_idx = rw.AddAtom(Chem.Atom(9))
    rw.AddBond(idx, f_idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return Chem.Mol(rw)
    except Exception:
        return None


def _add_methyl(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    candidates = []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0 and not a.GetIsAromatic():
            candidates.append(a.GetIdx())
        elif a.GetAtomicNum() == 7 and a.GetTotalNumHs() > 0:
            candidates.append(a.GetIdx())
    if not candidates:
        return None
    idx = random.choice(candidates)
    c_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(idx, c_idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return Chem.Mol(rw)
    except Exception:
        return None


def _replace_ch_with_n(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    candidates = [
        a.GetIdx() for a in rw.GetAtoms()
        if a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
    ]
    if not candidates:
        return None
    idx = random.choice(candidates)
    rw.GetAtomWithIdx(idx).SetAtomicNum(7)
    rw.GetAtomWithIdx(idx).SetNoImplicit(True)
    try:
        Chem.SanitizeMol(rw)
        return Chem.Mol(rw)
    except Exception:
        return None


def _add_hydroxyl(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    candidates = [
        a.GetIdx() for a in rw.GetAtoms()
        if not a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
    ]
    if not candidates:
        return None
    idx = random.choice(candidates)
    o_idx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(idx, o_idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return Chem.Mol(rw)
    except Exception:
        return None


_MUTATORS = [_add_fluorine, _add_methyl, _replace_ch_with_n, _add_hydroxyl]


def _mutate(mol: Chem.Mol, n_mutations: int = 2) -> Optional[Chem.Mol]:
    current = mol
    for _ in range(n_mutations):
        mutator = random.choice(_MUTATORS)
        result = mutator(current)
        if result is not None:
            current = result
    return current


# ── Scoring helpers ───────────────────────────────────────────────────────────

def compute_sa_score(mol: Chem.Mol) -> float:
    if HAS_SA_SCORE:
        try:
            return round(sascorer.calculateScore(mol), 3)
        except Exception:
            pass
    n_rings = Chem.GetSSSR(mol)
    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    return round(min(10.0, max(1.0, 1.0 + n_rings * 0.6 + n_atoms * 0.05 + n_chiral * 0.5)), 3)


def passes_lipinski(mol: Chem.Mol) -> bool:
    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if Lipinski.NumHDonors(mol) > 5:
        violations += 1
    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1
    return violations <= 1


def is_pains(mol: Chem.Mol) -> bool:
    return _PAINS_CATALOG.HasMatch(mol)


def _composite_score(qed_val: float, sa_val: float) -> float:
    return qed_val / (sa_val / 5.0)


# ── Minimum size check ───────────────────────────────────────────────────────

MIN_HEAVY_ATOMS = 8   # filter out trivially small molecules like fluorobenzene


# ── Public API ────────────────────────────────────────────────────────────────

def generate_molecules(pharmacophore_hints: str = "", n: int = 20) -> List[Dict]:
    """
    Generate *n* novel drug-like molecules.
    Returns a list of dicts with keys:
        ``smiles``, ``mol``, ``qed``, ``sa_score``, ``source``
    """
    raw_smiles: set[str] = set()

    # ── Strategy 1: mono-scaffold + R-group ──────────────────────────────
    for _ in range(n):
        scaff = random.choice(MONO_SCAFFOLDS)
        rg = random.choice(R_GROUPS)
        mol = _combine_at_dummy(scaff, [rg])
        if mol:
            smi = Chem.MolToSmiles(mol)
            raw_smiles.add(smi)

    # ── Strategy 2: di-scaffold + 2 R-groups ─────────────────────────────
    for _ in range(n // 2):
        scaff = random.choice(DI_SCAFFOLDS)
        rg1 = random.choice(R_GROUPS)
        rg2 = random.choice(R_GROUPS)
        mol = _combine_at_dummy(scaff, [rg1, rg2])
        if mol:
            smi = Chem.MolToSmiles(mol)
            raw_smiles.add(smi)

    # ── Strategy 3: mutate seed molecules ────────────────────────────────
    for seed_smi in SEED_MOLECULES:
        seed_mol = Chem.MolFromSmiles(seed_smi)
        if seed_mol is None:
            continue
        for _ in range(3):
            mutant = _mutate(seed_mol, n_mutations=random.randint(1, 3))
            if mutant:
                raw_smiles.add(Chem.MolToSmiles(mutant))

    logger.info("Generated %d unique raw SMILES.", len(raw_smiles))

    # ── Validate and score ───────────────────────────────────────────────
    molecules: list[dict] = []
    for smi in raw_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue

        # Filter out trivially small molecules
        heavy_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        if heavy_count < MIN_HEAVY_ATOMS:
            continue

        molecules.append({
            "smiles": Chem.MolToSmiles(mol),
            "mol": mol,
            "qed": round(qed(mol), 4),
            "sa_score": compute_sa_score(mol),
            "source": "generated",
        })

    logger.info("Validated %d molecules (≥ %d heavy atoms).", len(molecules), MIN_HEAVY_ATOMS)
    return molecules


def filter_molecules(
    molecules: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Apply drug-likeness filters and return top-k candidates.
    """
    filtered: list[dict] = []
    for m in molecules:
        mol = m["mol"]
        if not passes_lipinski(mol):
            continue
        if is_pains(mol):
            continue
        if m["qed"] < 0.3:
            continue
        if m["sa_score"] > 7.0:
            continue
        m["composite_score"] = round(_composite_score(m["qed"], m["sa_score"]), 4)
        filtered.append(m)

    filtered.sort(key=lambda x: x["composite_score"], reverse=True)
    top = filtered[:top_k]

    logger.info(
        "Filtered %d → %d molecules; returning top %d.",
        len(molecules), len(filtered), len(top),
    )
    return top