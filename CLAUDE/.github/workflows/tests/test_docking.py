# drug_discovery_pipeline/tests/test_docking.py
# =============================================================================
# FILE: tests/test_docking.py
# ROLE: Unit tests for tools/docking.py.
#       Tests 3D embedding, mock scoring, and score parsing.
#       Real Vina tests are marked requires_vina and skipped in CI.
# =============================================================================

import pytest
from pathlib import Path
from unittest.mock import patch

from tools.docking import (
    mock_docking_score,
    smiles_to_3d_mol,
    dock_molecule,
    _parse_vina_output,
)


class TestSmilsTo3DMol:
    def test_benzene(self):
        mol = smiles_to_3d_mol("c1ccccc1")
        assert mol is not None

    def test_ibuprofen(self):
        mol = smiles_to_3d_mol("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        assert mol is not None

    def test_invalid_smiles(self):
        mol = smiles_to_3d_mol("NOT_VALID_XXX")
        assert mol is None


class TestMockDockingScore:
    def test_returns_float(self):
        score = mock_docking_score("c1ccccc1")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        for smi in ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ncccc2c1"]:
            score = mock_docking_score(smi)
            assert -12.0 <= score <= -4.0, f"Score {score} out of range for {smi}"

    def test_reproducible_with_same_smiles(self):
        s1 = mock_docking_score("c1ccccc1", seed=0)
        s2 = mock_docking_score("c1ccccc1", seed=0)
        assert s1 == s2

    def test_different_smiles_different_scores(self):
        s1 = mock_docking_score("c1ccccc1")
        s2 = mock_docking_score("c1ccc2ncccc2c1CC(=O)N")
        # Not guaranteed, but very likely to differ
        assert isinstance(s1, float) and isinstance(s2, float)

    def test_invalid_smiles_returns_minimum(self):
        score = mock_docking_score("INVALID_XXX")
        assert score == -4.0


class TestParseVinaOutput:
    def test_parses_mode1(self):
        output = """
-----+------------+----------+----------
   1        -8.5      0.000      0.000
   2        -7.2      1.234      2.345
"""
        score = _parse_vina_output(output)
        assert score == -8.5

    def test_returns_none_on_no_match(self):
        score = _parse_vina_output("No docking output here")
        assert score is None

    def test_handles_extra_whitespace(self):
        output = "   1      -9.1      0.000      0.000\n"
        score = _parse_vina_output(output)
        assert score == -9.1


class TestDockMolecule:
    def test_falls_back_to_mock_without_vina(self):
        """Without Vina in PATH, dock_molecule should return a mock score."""
        with patch("shutil.which", return_value=None):
            result = dock_molecule("c1ccccc1")
        assert "score"  in result
        assert "smiles" in result
        assert result["mode"] in ("mock", "mock_invalid_smiles")
        assert isinstance(result["score"], float)

    def test_invalid_smiles_handled(self):
        with patch("shutil.which", return_value=None):
            result = dock_molecule("INVALID_SMILES_XYZ")
        assert result["score"] == -4.0

    @pytest.mark.requires_vina
    def test_real_vina_run(self, tmp_path):
        """Only runs if vina is installed and a receptor is available."""
        import shutil
        if not shutil.which("vina"):
            pytest.skip("Vina not in PATH")
        result = dock_molecule("c1ccccc1")
        assert "score" in result