# drug_discovery_pipeline/tests/test_config.py
# =============================================================================
# FILE: tests/test_config.py
# ROLE: Tests for config.py — verifies all expected settings exist and
#       have sensible defaults.
# =============================================================================

import pytest
import config


class TestConfig:
    def test_directories_exist_as_path_objects(self):
        from pathlib import Path
        assert isinstance(config.BASE_DIR,   Path)
        assert isinstance(config.OUTPUT_DIR, Path)
        assert isinstance(config.WORK_DIR,   Path)
        assert isinstance(config.CHROMA_DIR, Path)

    def test_ollama_urls_are_strings(self):
        assert isinstance(config.LOCAL_OLLAMA_URL,  str)
        assert isinstance(config.REMOTE_OLLAMA_URL, str)
        assert config.LOCAL_OLLAMA_URL.startswith("http")

    def test_task_model_map_has_required_tasks(self):
        required = [
            "plan", "hypothesis", "pharmacophore", "molecule_refine",
            "docking_interpret", "retrosynthesis", "executive_summary",
            "literature_synthesis", "json_repair", "section_polish",
            "default",
        ]
        for task in required:
            assert task in config.TASK_MODEL_MAP, f"Missing task: {task}"

    def test_task_model_map_values_are_tuples(self):
        for task, val in config.TASK_MODEL_MAP.items():
            assert isinstance(val, tuple), f"{task} should map to a tuple"
            assert len(val) == 2,          f"{task} tuple should have 2 elements"
            model, server = val
            assert isinstance(model,  str)
            assert isinstance(server, str)
            assert server.startswith("http")

    def test_lipinski_thresholds_are_positive(self):
        assert config.LIPINSKI_MW_MAX   > 0
        assert config.LIPINSKI_LOGP_MAX > 0
        assert config.LIPINSKI_HBD_MAX  > 0
        assert config.LIPINSKI_HBA_MAX  > 0

    def test_candidate_counts_are_sane(self):
        assert config.NUM_CANDIDATES_GENERATE  >= config.NUM_CANDIDATES_SHORTLIST
        assert config.NUM_CANDIDATES_SHORTLIST  > 0

    def test_llm_timeout_positive(self):
        assert config.LLM_TIMEOUT > 0

    def test_remote_model_name(self):
        assert "gemma4" in config.REMOTE_LLM_MODEL

    def test_embedding_model_set(self):
        assert len(config.EMBEDDING_MODEL) > 0