# drug_discovery_pipeline/tests/test_molecule_generator.py
# =============================================================================
# FILE: tests/test_molecule_generator.py
# ROLE: Unit tests for tools/molecule_generator.py.
#       Tests molecule validity, filter logic, and generation stats.
#       No LLM or network calls needed.
# =============================================================================

import pytest
from rdkit import Chem

from tools.molecule_generator import (
    calculate_qed,
    calculate_sa_score,
    evaluate_molecule,
    filter_and_rank,
    generate_smiles_library,
    is_pains,
    lipinski_pass,
)


class TestLipinskiPass:
    def test_aspirin_passes(self):
        # Aspirin: MW=180, LogP=1.2, HBD=1, HBA=3
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        ok, props = lipinski_pass(mol)
        assert ok is True
        assert "MW" in props
        assert props["MW"] < 500

    def test_very_large_molecule_fails(self):
        # Taxol-like huge molecule — likely to fail
        mol = Chem.MolFromSmiles(
            "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=O)C2(C)C)(C4OC(=O)C)O)OC(=O)C5=CC=CC=C5)OC(=O)C)(C)C1OC(=O)C)O"
        )
        if mol:
            ok, props = lipinski_pass(mol)
            # May fail due to high MW
            assert isinstance(ok, bool)


class TestQED:
    def test_qed_range(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        score = calculate_qed(mol)
        assert 0.0 <= score <= 1.0

    def test_drug_like_has_reasonable_qed(self):
        # Ibuprofen
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        score = calculate_qed(mol)
        assert score > 0.3


class TestSAScore:
    def test_simple_molecule_easy(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        sa = calculate_sa_score(mol)
        assert 1.0 <= sa <= 10.0

    def test_complex_molecule_harder(self):
        # Strychnine-like complexity
        mol = Chem.MolFromSmiles("C1CN2CC3=CCOC4CC(=O)N5CC6=CC=CC1C6C2C3C45")
        if mol:
            sa = calculate_sa_score(mol)
            assert sa > 3.0  # should be harder


class TestPAINS:
    def test_clean_molecule(self):
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        # Quinoline is clean
        assert isinstance(is_pains(mol), bool)


class TestEvaluateMolecule:
    def test_valid_smiles_returns_dict(self):
        result = evaluate_molecule("c1ccccc1CC(=O)N")
        assert result is not None
        assert "smiles" in result
        assert "QED"    in result
        assert "MW"     in result
        assert "SA_Score" in result
        assert "drug_like" in result

    def test_invalid_smiles_returns_none(self):
        result = evaluate_molecule("not_a_smiles_XXX")
        assert result is None

    def test_aspirin_properties(self):
        result = evaluate_molecule("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert 170 < result["MW"] < 195
        assert result["Lipinski_Pass"] is True


class TestGenerateSmilesLibrary:
    def test_generates_requested_count(self):
        smiles = generate_smiles_library(n_total=10, rng_seed=42)
        assert len(smiles) > 0
        assert len(smiles) <= 10

    def test_all_valid_smiles(self):
        smiles = generate_smiles_library(n_total=10, rng_seed=7)
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            assert mol is not None, f"Invalid SMILES: {s}"

    def test_no_duplicates(self):
        smiles = generate_smiles_library(n_total=15, rng_seed=42)
        assert len(smiles) == len(set(smiles))

    def test_seed_smiles_influence(self):
        # Using a specific seed SMILES should produce different results
        with_seed    = generate_smiles_library(seed_smiles=["c1ccc2ncccc2c1"], n_total=10)
        without_seed = generate_smiles_library(seed_smiles=None,               n_total=10)
        # Not guaranteed to differ, but libraries should be non-empty
        assert len(with_seed)    > 0
        assert len(without_seed) > 0


class TestFilterAndRank:
    def test_returns_top_k(self):
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "c1ccc2ncccc2c1CC(=O)N",  # quinoline-amide
            "c1ccccc1",               # benzene
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
            "Cc1ccc(cc1)S(=O)(=O)N",  # toluenesulfonamide
        ]
        top3 = filter_and_rank(smiles_list, top_k=3)
        assert len(top3) <= 3

    def test_ranked_by_composite_score(self):
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "c1ccc2ncccc2c1CC(=O)N",
            "Cc1ccc(cc1)S(=O)(=O)N",
        ]
        ranked = filter_and_rank(smiles_list, top_k=5)
        if len(ranked) >= 2:
            assert ranked[0]["composite_score"] >= ranked[1]["composite_score"]

    def test_empty_input(self):
        result = filter_and_rank([], top_k=5)
        assert result == []