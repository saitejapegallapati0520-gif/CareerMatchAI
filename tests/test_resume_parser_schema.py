"""Pytest tests for resume parser output schema (no real Gemini API calls)."""

from __future__ import annotations

import pytest

from Modules.resume_parser import (
    clean_text,
    extract_resume_info,
    extract_keywords,
    parse_resume,
)


STRUCTURED_INFO_KEYS = [
    "name",
    "email",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
    "summary",
    "top_ats_keywords",
]


def test_clean_text_returns_string() -> None:
    """clean_text returns a string."""
    raw = "John Doe\njohn@email.com\nSkills: Python, SQL\nExperience: 2 years backend."
    result = clean_text(raw)
    assert isinstance(result, str)


def test_extract_resume_info_empty_text_returns_full_schema() -> None:
    """extract_resume_info('') returns dict with all required keys and correct types."""
    result = extract_resume_info("")
    assert isinstance(result, dict)
    for key in STRUCTURED_INFO_KEYS:
        assert key in result, f"missing key: {key}"
    assert isinstance(result["name"], str)
    assert isinstance(result["email"], str)
    assert isinstance(result["summary"], str)
    assert isinstance(result["skills"], list)
    assert isinstance(result["experience"], list)
    assert isinstance(result["education"], list)
    assert isinstance(result["projects"], list)
    assert isinstance(result["certifications"], list)
    assert isinstance(result["top_ats_keywords"], list)
    assert all(isinstance(x, str) for x in result["skills"])
    assert all(isinstance(x, str) for x in result["experience"])


def test_extract_keywords_empty_text_returns_list() -> None:
    """extract_keywords('') returns a list."""
    result = extract_keywords("")
    assert isinstance(result, list)
    assert result == []


def test_parse_resume_invalid_path_returns_full_schema() -> None:
    """parse_resume(nonexistent path) returns dict with raw_text, structured_info, keywords."""
    result = parse_resume("/nonexistent/path/resume.pdf")
    assert isinstance(result, dict)
    assert "raw_text" in result
    assert "structured_info" in result
    assert "keywords" in result
    assert result["raw_text"] == ""
    assert isinstance(result["keywords"], list)
    info = result["structured_info"]
    assert isinstance(info, dict)
    for key in STRUCTURED_INFO_KEYS:
        assert key in info, f"structured_info missing key: {key}"
    assert isinstance(info["name"], str)
    assert isinstance(info["email"], str)
    assert isinstance(info["summary"], str)
    assert isinstance(info["skills"], list)
    assert isinstance(info["experience"], list)
    assert isinstance(info["education"], list)
    assert isinstance(info["projects"], list)
    assert isinstance(info["certifications"], list)
    assert isinstance(info["top_ats_keywords"], list)


@pytest.mark.parametrize("key", STRUCTURED_INFO_KEYS)
def test_parse_resume_fallback_structured_info_has_key(key: str) -> None:
    """structured_info from parse_resume fallback contains each required key."""
    result = parse_resume("/nonexistent/resume.pdf")
    assert key in result["structured_info"]


def test_extract_resume_info_with_mocked_llm_returns_valid_schema() -> None:
    """With mocked LLM, extract_resume_info returns valid schema (str/list types)."""
    from unittest.mock import patch

    mock_json = (
        '{"name": "John Doe", "email": "john@email.com", "skills": ["Python", "SQL"], '
        '"experience": ["2 years backend"], "education": [], "projects": [], '
        '"certifications": [], "summary": "Backend developer.", "top_ats_keywords": ["python", "sql"]}'
    )

    with patch("Modules.resume_parser._invoke_with_fallback", return_value=mock_json):
        raw = "John Doe\njohn@email.com\nSkills: Python, SQL\nExperience: 2 years backend."
        result = extract_resume_info(raw)
    assert isinstance(result, dict)
    for key in STRUCTURED_INFO_KEYS:
        assert key in result
    assert isinstance(result["name"], str)
    assert isinstance(result["email"], str)
    assert isinstance(result["summary"], str)
    assert isinstance(result["skills"], list)
    assert isinstance(result["experience"], list)
    assert all(isinstance(x, str) for x in result["skills"])
    assert all(isinstance(x, str) for x in result["experience"])


def test_extract_keywords_with_mocked_llm_returns_list_of_strings() -> None:
    """With mocked LLM, extract_keywords returns a list of strings."""
    from unittest.mock import patch

    with patch("Modules.resume_parser._invoke_with_fallback", return_value='{"keywords": ["python", "sql", "backend"]}'):
        raw = "John Doe\nSkills: Python, SQL\nExperience: 2 years backend."
        result = extract_keywords(raw)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
