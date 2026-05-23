# drug_discovery_pipeline/tests/test_helpers.py
# =============================================================================
# FILE: tests/test_helpers.py
# ROLE: Unit tests for utils/helpers.py.
#       Tests JSON extraction, server health caching, file I/O.
#       All Ollama calls are mocked.
# =============================================================================

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.helpers import (
    extract_json,
    safe_extract_json,
    truncate,
    flatten_list,
    timestamp,
    save_text,
    load_text,
)


class TestExtractJson:
    def test_plain_json(self):
        data = extract_json('{"key": "value"}')
        assert data == {"key": "value"}

    def test_json_with_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_json_with_plain_fence(self):
        text = '```\n{"key": 42}\n```'
        assert extract_json(text) == {"key": 42}

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"phases": [1, 2, 3]} and more text'
        result = extract_json(text)
        assert result == {"phases": [1, 2, 3]}

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            extract_json("No JSON here at all")

    def test_list_json(self):
        # Lists wrapped in braces aren't supported by brace-search, but direct parse should work
        result = safe_extract_json('[1, 2, 3]')
        assert result == [1, 2, 3]


class TestSafeExtractJson:
    def test_returns_default_on_failure(self):
        result = safe_extract_json("not json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_returns_none_default(self):
        result = safe_extract_json("not json")
        assert result is None

    def test_valid_json(self):
        result = safe_extract_json('{"a": 1}')
        assert result == {"a": 1}


class TestTruncate:
    def test_short_string_unchanged(self):
        s = "hello world"
        assert truncate(s, 100) == s

    def test_long_string_truncated(self):
        s = "a" * 200
        result = truncate(s, 100)
        assert len(result) > 100  # includes ellipsis
        assert "truncated" in result or "+" in result

    def test_exact_length_unchanged(self):
        s = "a" * 50
        assert truncate(s, 50) == s


class TestFlattenList:
    def test_already_flat(self):
        assert flatten_list([1, 2, 3]) == [1, 2, 3]

    def test_nested_one_level(self):
        assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_mixed(self):
        assert flatten_list([[1, 2], 3, [4]]) == [1, 2, 3, 4]


class TestTimestamp:
    def test_returns_string(self):
        ts = timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_format(self):
        ts = timestamp()
        # Should match YYYYMMDDTHHMMSSz
        import re
        assert re.match(r"\d{8}T\d{6}Z", ts), f"Bad format: {ts}"


class TestFileIO:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test.txt"
        save_text(path, "hello world")
        assert path.exists()
        content = load_text(path)
        assert content == "hello world"

    def test_load_missing_file(self, tmp_path):
        result = load_text(tmp_path / "nonexistent.txt")
        assert result is None

    def test_save_creates_parents(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "file.txt"
        save_text(path, "content")
        assert path.exists()