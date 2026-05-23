# drug_discovery_pipeline/tests/conftest.py
# =============================================================================
# FILE: tests/conftest.py
# ROLE: Shared pytest fixtures and marks.
#       Marks tests that require a live Ollama server, Vina, or network
#       so they can be skipped in CI.
# =============================================================================

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Custom pytest marks
# ---------------------------------------------------------------------------
# Register marks to avoid PytestUnknownMarkWarning
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_ollama: test needs a live Ollama server"
    )
    config.addinivalue_line(
        "markers", "requires_vina: test needs AutoDock Vina in PATH"
    )
    config.addinivalue_line(
        "markers", "requires_network: test makes real HTTP requests"
    )


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, tmp_path):
    """
    Set safe environment variables for every test.
    Redirects output dirs to tmp_path so tests don't pollute the repo.
    """
    monkeypatch.setenv("LOCAL_OLLAMA_URL",  "http://localhost:11434/v1")
    monkeypatch.setenv("REMOTE_OLLAMA_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("LOG_LEVEL",         "WARNING")
    monkeypatch.setenv("RECEPTOR_PDBQT",    "")
    monkeypatch.setenv("NCBI_API_KEY",      "")
    # Redirect dirs
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("WORK_DIR",   str(tmp_path / "workdir"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma_db"))
    (tmp_path / "outputs").mkdir()
    (tmp_path / "workdir").mkdir()
    (tmp_path / "chroma_db").mkdir()


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm_call():
    """
    Patch routed_llm_call and llm_call to return deterministic JSON strings
    without needing a live Ollama server.
    """
    with patch("utils.helpers.routed_llm_call") as mock_routed, \
         patch("utils.helpers.llm_call") as mock_plain:

        def side_effect(task, messages, **kwargs):
            # Return minimal valid JSON for each task type
            responses = {
                "plan": '{"phases": [{"phase_number": 1, "phase_name": "Test Phase", "description": "desc", "expected_outputs": ["output1"]}]}',
                "hypothesis": '{"selected_target": {"gene_name": "EGFR", "uniprot_id": "P00533", "pdb_id": "1IVO", "rationale": "Test rationale"}, "hypothesis": "Test hypothesis.", "justification": "Test justification.", "druggability_score": 0.8, "confidence": "high"}',
                "pharmacophore": '{"pharmacophore_features": ["H-bond donor"], "core_scaffolds": ["c1ccccc1"], "forbidden_groups": [], "design_strategy": "Test strategy"}',
                "molecule_refine": '{"top_candidates": [{"smiles": "c1ccccc1", "rationale": "test", "predicted_activity": "medium"}], "design_notes": "test notes"}',
                "docking_interpret": '{"ranked_compounds": [], "best_candidate": "c1ccccc1", "structural_insights": "Test insight."}',
                "retrosynthesis": '{"retrosynthetic_steps": ["step1"], "forward_route": [{"step": 1, "reaction": "test", "reagents": "test", "conditions": "test", "expected_yield_percent": 70}], "starting_materials": ["benzene"], "estimated_steps": 3, "feasibility": "moderate", "key_challenges": []}',
                "literature_synthesis": '{"key_targets": ["EGFR"], "known_mechanisms": ["kinase inhibition"], "known_compounds": ["erlotinib"], "research_gaps": ["selectivity"], "summary": "Test summary."}',
                "executive_summary": "This is a test executive summary for the drug discovery project.",
                "section_polish": "Polished section text.",
                "conclusion": "Test conclusion.",
                "json_repair": "{}",
                "default": "{}",
            }
            return responses.get(task, '{"result": "mock"}')

        mock_routed.side_effect = side_effect
        mock_plain.return_value = '{"result": "mock"}'
        yield mock_routed, mock_plain


@pytest.fixture
def mock_embedding():
    """Return a fixed 768-dim embedding vector."""
    with patch("utils.helpers.get_embedding") as mock:
        mock.return_value = [0.01] * 768
        with patch("utils.helpers.get_embeddings_batch") as mock_batch:
            mock_batch.return_value = [[0.01] * 768]
            yield mock, mock_batch


@pytest.fixture
def mock_network():
    """Block all real HTTP requests in unit tests."""
    with patch("requests.get")  as mock_get, \
         patch("requests.post") as mock_post:
        mock_get.return_value  = MagicMock(status_code=200, json=lambda: {}, text="")
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {}, text="")
        yield mock_get, mock_post


@pytest.fixture
def sample_state():
    """Minimal pipeline state dict for agent unit tests."""
    return {
        "run_id":    "test_run_001",
        "input":     "EGFR non-small cell lung cancer",
        "started_at": "20240101T000000Z",
        "plan":      [],
        "literature": {
            "abstracts": ["EGFR is a key target in NSCLC..."],
            "synthesis": {
                "key_targets":      ["EGFR", "ALK"],
                "known_mechanisms": ["kinase inhibition"],
                "known_compounds":  ["erlotinib", "gefitinib"],
                "research_gaps":    ["T790M resistance"],
                "summary":          "EGFR inhibition is validated in NSCLC.",
            },
            "count": 1,
        },
        "patents": {
            "results":        [],
            "chemical_claims": ["c1ccc2ncccc2c1"],
            "patent_count":   5,
        },
        "target_info": {
            "gene_name":      "EGFR",
            "uniprot_id":     "P00533",
            "pdb_id":         "1IVO",
            "protein_name":   "Epidermal growth factor receptor",
            "function":       "Receptor tyrosine kinase",
            "diseases":       ["Non-small cell lung carcinoma"],
            "pdb_local_path": "",
        },
        "hypothesis": {
            "selected_target": {
                "gene_name":  "EGFR",
                "uniprot_id": "P00533",
                "pdb_id":     "1IVO",
                "rationale":  "Well-validated oncology target.",
            },
            "hypothesis":         "Selective EGFR inhibition blocks tumour proliferation.",
            "justification":      "EGFR is mutated in 15% of NSCLC.",
            "druggability_score": 0.9,
            "confidence":         "high",
            "pdb_local_path":     "",
        },
        "pharmacophore": {
            "pharmacophore_features": ["H-bond donor", "aromatic ring"],
            "core_scaffolds":   ["c1ccc2ncccc2c1"],
            "forbidden_groups": ["aldehyde"],
            "design_strategy":  "Fragment-based from quinoline scaffold",
        },
        "candidates": {
            "shortlist": [
                {
                    "smiles":     "c1ccc2ncccc2c1CC(=O)N",
                    "MW":         188.2,
                    "LogP":       1.8,
                    "HBD":        1,
                    "HBA":        2,
                    "TPSA":       38.3,
                    "QED":        0.72,
                    "SA_Score":   2.1,
                    "drug_like":  True,
                    "PAINS":      False,
                    "composite_score": 0.615,
                },
            ],
            "all_candidates": [],
            "generation_stats": {
                "total_generated": 20,
                "valid_molecules": 15,
                "drug_like":       8,
                "shortlisted":     1,
            },
        },
        "docking_results": [
            {
                "smiles":       "c1ccc2ncccc2c1CC(=O)N",
                "score":        -8.2,
                "mode":         "mock",
                "docking_mode": "mock",
                "QED":          0.72,
                "MW":           188.2,
                "LogP":         1.8,
                "SA_Score":     2.1,
            },
        ],
        "docking_interpretation": {
            "best_candidate":    "c1ccc2ncccc2c1CC(=O)N",
            "structural_insights": "The compound fits the ATP pocket.",
            "ranked_compounds":  [],
        },
        "synthesis": [
            {
                "smiles":      "c1ccc2ncccc2c1CC(=O)N",
                "sa_score":    2.1,
                "sa_category": "easy",
                "feasibility": "straightforward",
                "estimated_steps": 3,
                "rule_based_disconnections": ["  → Amide bond hydrolysis"],
                "llm_route": {
                    "forward_route": [
                        {
                            "step": 1, "reaction": "Amide coupling",
                            "reagents": "HATU, DIPEA", "conditions": "DMF, RT",
                            "expected_yield_percent": 75,
                        }
                    ],
                    "key_challenges": [],
                    "estimated_steps": 3,
                    "feasibility": "straightforward",
                },
            }
        ],
        "report":      "",
        "report_path": "",
        "errors":      [],
        "metadata":    {"run_id": "test_run_001"},
    }