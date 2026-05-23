# drug_discovery_pipeline/tests/test_agents.py
# =============================================================================
# FILE: tests/test_agents.py
# ROLE: Integration-style tests for agents.
#       All external calls (LLM, HTTP) are mocked.
#       Tests the run() method of each agent using sample_state fixture.
# =============================================================================

import pytest
from unittest.mock import patch, MagicMock


class TestHypothesisAgent:
    def test_run_populates_hypothesis(self, sample_state, mock_llm_call):
        from agents.hypothesis import HypothesisAgent
        agent  = HypothesisAgent()
        result = agent.run(sample_state)
        assert "hypothesis" in result
        assert "selected_target" in result["hypothesis"]
        assert isinstance(result["hypothesis"]["selected_target"], dict)

    def test_run_uses_fallback_on_bad_llm(self, sample_state):
        with patch("utils.helpers.routed_llm_call", return_value="not valid json"):
            from agents.hypothesis import HypothesisAgent
            agent  = HypothesisAgent()
            result = agent.run(sample_state)
            assert "hypothesis" in result
            # Fallback should still produce a gene_name
            assert "gene_name" in result["hypothesis"]["selected_target"]


class TestMoleculeDesignerAgent:
    def test_run_produces_candidates(self, sample_state, mock_llm_call):
        from agents.molecule_designer import MoleculeDesignerAgent
        agent  = MoleculeDesignerAgent()
        result = agent.run(sample_state)
        assert "candidates" in result
        assert "shortlist"  in result["candidates"]
        assert "generation_stats" in result["candidates"]
        stats = result["candidates"]["generation_stats"]
        assert stats["total_generated"] > 0

    def test_run_produces_pharmacophore(self, sample_state, mock_llm_call):
        from agents.molecule_designer import MoleculeDesignerAgent
        agent  = MoleculeDesignerAgent()
        result = agent.run(sample_state)
        assert "pharmacophore" in result
        assert "pharmacophore_features" in result["pharmacophore"]


class TestDockingEvaluatorAgent:
    def test_run_with_no_shortlist(self, sample_state, mock_llm_call):
        from agents.docking_evaluator import DockingEvaluatorAgent
        sample_state["candidates"] = {}
        agent  = DockingEvaluatorAgent()
        result = agent.run(sample_state)
        assert result["docking_results"] == []

    def test_run_produces_docking_results(self, sample_state, mock_llm_call):
        with patch("tools.docking.dock_multiple") as mock_dock:
            mock_dock.return_value = [{
                "smiles": "c1ccc2ncccc2c1CC(=O)N",
                "score":  -8.2,
                "mode":   "mock",
            }]
            from agents.docking_evaluator import DockingEvaluatorAgent
            agent  = DockingEvaluatorAgent()
            result = agent.run(sample_state)
            assert len(result["docking_results"]) > 0
            assert "score" in result["docking_results"][0]


class TestSynthesisEvaluatorAgent:
    def test_run_produces_synthesis(self, sample_state, mock_llm_call):
        from agents.synthesis_evaluator import SynthesisEvaluatorAgent
        agent  = SynthesisEvaluatorAgent()
        result = agent.run(sample_state)
        assert "synthesis" in result
        assert isinstance(result["synthesis"], list)

    def test_run_empty_docking_results(self, sample_state, mock_llm_call):
        sample_state["docking_results"] = []
        from agents.synthesis_evaluator import SynthesisEvaluatorAgent
        agent  = SynthesisEvaluatorAgent()
        result = agent.run(sample_state)
        assert result["synthesis"] == []


class TestReportCompilerAgent:
    def test_run_produces_report(self, sample_state, mock_llm_call):
        from agents.report_compiler import ReportCompilerAgent
        agent  = ReportCompilerAgent()
        result = agent.run(sample_state)
        assert "report"      in result
        assert "report_path" in result
        assert len(result["report"]) > 100
        assert "# Drug Discovery" in result["report"]

    def test_report_contains_required_sections(self, sample_state, mock_llm_call):
        from agents.report_compiler import ReportCompilerAgent
        agent  = ReportCompilerAgent()
        result = agent.run(sample_state)
        report = result["report"]
        for section in [
            "Executive Summary",
            "Target Selection",
            "Molecule Design",
            "Docking",
        ]:
            assert section in report, f"Missing section: {section}"

    def test_report_file_created(self, sample_state, mock_llm_call, tmp_path):
        import config
        # Point config output dir to tmp
        original = config.OUTPUT_DIR
        config.OUTPUT_DIR = tmp_path
        try:
            from agents.report_compiler import ReportCompilerAgent
            agent  = ReportCompilerAgent()
            result = agent.run(sample_state)
            from pathlib import Path
            assert Path(result["report_path"]).exists()
        finally:
            config.OUTPUT_DIR = original