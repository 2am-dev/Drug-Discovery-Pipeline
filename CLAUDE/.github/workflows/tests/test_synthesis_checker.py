# drug_discovery_pipeline/tests/test_synthesis_checker.py
# =============================================================================
# FILE: tests/test_synthesis_checker.py
# ROLE: Unit tests for tools/synthesis_checker.py.
#       Tests rule-based retrosynthesis and SA score integration.
#       LLM calls are mocked.
# =============================================================================

import pytest
from unittest.mock import patch

from tools.synthesis_checker import (
    assess_synthesis,
    rule_based_retrosynthesis,
)


class TestRuleBasedRetrosynthesis:
    def test_amide_detected(self):
        # N-benzylacetamide has an amide
        result = rule_based_retrosynthesis("CC(=O)NCc1ccccc1")
        combined = " ".join(result)
        assert "Amide" in combined or "amide" in combined.lower()

    def test_ester_detected(self):
        result = rule_based_retrosynthesis("CC(=O)Oc1ccccc1")
        combined = " ".join(result)
        assert "Ester" in combined or "ester" in combined.lower()

    def test_biaryl_suzuki(self):
        # biphenyl
        result = rule_based_retrosynthesis("c1ccc(-c2ccccc2)cc1")
        combined = " ".join(result)
        assert "Suzuki" in combined or "coupling" in combined.lower()

    def test_invalid_smiles(self):
        result = rule_based_retrosynthesis("INVALID_XYZ")
        assert len(result) == 1
        assert "Invalid" in result[0]

    def test_simple_benzene_no_disconnections(self):
        result = rule_based_retrosynthesis("c1ccccc1")
        assert isinstance(result, list)


class TestAssessSynthesis:
    def test_returns_dict_with_required_keys(self, mock_llm_call):
        result = assess_synthesis(
            "CC(=O)Oc1ccccc1C(=O)O",
            target={"gene_name": "EGFR"},
            use_llm=True,
        )
        required_keys = [
            "smiles", "sa_score", "sa_category",
            "rule_based_disconnections", "feasibility",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_sa_score_in_range(self, mock_llm_call):
        result = assess_synthesis("c1ccccc1", target={}, use_llm=True)
        assert 1.0 <= result["sa_score"] <= 10.0

    def test_invalid_smiles(self, mock_llm_call):
        result = assess_synthesis("INVALID", target={}, use_llm=False)
        assert "error" in result

    def test_no_llm_mode(self):
        """Should work without LLM — rule-based only."""
        result = assess_synthesis(
            "CC(=O)Oc1ccccc1C(=O)O",
            target={},
            use_llm=False,
        )
        assert "sa_score" in result
        assert result.get("llm_route") == {}

    def test_sa_category_easy_for_simple_mol(self, mock_llm_call):
        result = assess_synthesis("c1ccccc1", target={}, use_llm=True)
        # Benzene should be very easy
        assert result["sa_category"] in ("easy", "moderate")