"""Pytest tests for matcher rank_jobs output (mock embeddings, no real Gemini API calls)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from Modules.matcher import rank_jobs


# Embedding dimension used by mock (must match for cosine_similarity).
MOCK_EMBED_DIM = 768


def _mock_embed_text(_text: str) -> list[float]:
    """Return a fixed-dimension vector so rank_jobs can compute cosine similarity without API."""
    return [0.1] * MOCK_EMBED_DIM


def _synthetic_resume_data() -> dict:
    """Minimal resume_data as produced by parse_resume / used by rank_jobs."""
    return {
        "raw_text": "Python developer with SQL and AWS experience.",
        "structured_info": {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "skills": ["Python", "SQL", "AWS"],
            "experience": ["3 years backend"],
            "education": ["B.S. Computer Science"],
            "projects": [],
            "certifications": [],
            "summary": "Backend developer.",
            "top_ats_keywords": [],
        },
        "keywords": ["python", "sql", "aws", "backend", "api"],
    }


def _synthetic_formatted_jobs() -> list[dict]:
    """Minimal formatted job list (same schema as format_jobs output)."""
    return [
        {
            "title": "Python Backend Engineer",
            "company": "Acme Inc",
            "location": "Remote",
            "description": "Python, SQL, AWS, REST APIs. Build scalable services.",
            "salary": "100,000 - 140,000",
            "url": "https://example.com/job1",
            "date_posted": "2026-03-01",
        },
        {
            "title": "Data Engineer",
            "company": "DataCo",
            "location": "New York",
            "description": "SQL, ETL, Python, cloud. Data pipelines.",
            "salary": "Not disclosed",
            "url": "https://example.com/job2",
            "date_posted": "2026-03-02",
        },
        {
            "title": "Full Stack Developer",
            "company": "WebCo",
            "location": "Austin",
            "description": "JavaScript, React, Node. Some Python preferred.",
            "salary": "90,000 - 120,000",
            "url": "https://example.com/job3",
            "date_posted": "2026-03-03",
        },
    ]


@patch("Modules.matcher._embed_text", side_effect=_mock_embed_text)
def test_rank_jobs_returns_same_length_as_input(mock_embed: object) -> None:
    """len(ranked) == len(jobs_list)."""
    resume_data = _synthetic_resume_data()
    jobs = _synthetic_formatted_jobs()
    ranked = rank_jobs(resume_data, jobs)
    assert len(ranked) == len(jobs)


@patch("Modules.matcher._embed_text", side_effect=_mock_embed_text)
def test_rank_jobs_sorted_descending_by_score(mock_embed: object) -> None:
    """Scores are in descending order."""
    resume_data = _synthetic_resume_data()
    jobs = _synthetic_formatted_jobs()
    ranked = rank_jobs(resume_data, jobs)
    for i in range(len(ranked) - 1):
        assert ranked[i]["score"] >= ranked[i + 1]["score"], (
            f"ranked[{i}]['score']={ranked[i]['score']} < ranked[{i+1}]['score']={ranked[i+1]['score']}"
        )


@patch("Modules.matcher._embed_text", side_effect=_mock_embed_text)
def test_rank_jobs_each_item_has_score_key(mock_embed: object) -> None:
    """Each ranked item has numeric 'score' key."""
    resume_data = _synthetic_resume_data()
    jobs = _synthetic_formatted_jobs()
    ranked = rank_jobs(resume_data, jobs)
    for item in ranked:
        assert "score" in item
        assert isinstance(item["score"], (int, float))


@patch("Modules.matcher._embed_text", side_effect=_mock_embed_text)
def test_rank_jobs_empty_jobs_returns_empty_list(mock_embed: object) -> None:
    """rank_jobs(resume_data, []) returns []."""
    resume_data = _synthetic_resume_data()
    ranked = rank_jobs(resume_data, [])
    assert ranked == []


@patch("Modules.matcher._embed_text", side_effect=_mock_embed_text)
def test_rank_jobs_enriches_with_matched_missing_skills(mock_embed: object) -> None:
    """Ranked jobs include matched_skills and missing_skills."""
    resume_data = _synthetic_resume_data()
    jobs = _synthetic_formatted_jobs()
    ranked = rank_jobs(resume_data, jobs)
    for item in ranked:
        assert "matched_skills" in item
        assert "missing_skills" in item
        assert isinstance(item["matched_skills"], list)
        assert isinstance(item["missing_skills"], list)
