"""Pytest tests for job search format_jobs output schema (no real Adzuna API calls)."""

from __future__ import annotations

import pytest

from Modules.job_search import format_jobs


def _fake_raw_adzuna_job(
    title: str = "Software Engineer",
    company_name: str = "Acme Corp",
    location_name: str = "New York",
    description: str = "Build APIs.",
    salary_min: int | None = 80000,
    salary_max: int | None = 120000,
    redirect_url: str = "https://example.com/job/1",
    created: str = "2026-03-01",
) -> dict:
    """Build a single fake raw Adzuna-style job dict."""
    return {
        "title": title,
        "company": {"display_name": company_name},
        "location": {"display_name": location_name},
        "description": description,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "redirect_url": redirect_url,
        "created": created,
    }


REQUIRED_FORMATTED_KEYS = [
    "title",
    "company",
    "location",
    "description",
    "salary",
    "url",
    "date_posted",
]


def test_format_jobs_empty_list_returns_empty_list() -> None:
    """format_jobs([]) returns []."""
    result = format_jobs([])
    assert result == []


def test_format_jobs_single_job_has_all_required_keys() -> None:
    """Each formatted job has exactly title, company, location, description, salary, url, date_posted."""
    raw = [_fake_raw_adzuna_job()]
    result = format_jobs(raw)
    assert len(result) == 1
    job = result[0]
    for key in REQUIRED_FORMATTED_KEYS:
        assert key in job, f"missing key: {key}"
    assert set(job.keys()) == set(REQUIRED_FORMATTED_KEYS)


def test_format_jobs_salary_non_empty_string() -> None:
    """Formatted job salary is a non-empty string."""
    raw = [_fake_raw_adzuna_job(salary_min=100000, salary_max=150000)]
    result = format_jobs(raw)
    assert len(result) == 1
    assert isinstance(result[0]["salary"], str)
    assert len(result[0]["salary"].strip()) > 0


def test_format_jobs_salary_not_disclosed_when_no_min_max() -> None:
    """When salary_min/salary_max missing, salary is 'Not disclosed' or similar non-empty string."""
    raw = [_fake_raw_adzuna_job(salary_min=None, salary_max=None)]
    result = format_jobs(raw)
    assert len(result) == 1
    assert isinstance(result[0]["salary"], str)
    assert result[0]["salary"].strip() != ""


def test_format_jobs_url_and_date_posted_are_strings() -> None:
    """url and date_posted are strings."""
    raw = [_fake_raw_adzuna_job(redirect_url="https://apply.here", created="2026-03-05")]
    result = format_jobs(raw)
    assert len(result) == 1
    assert isinstance(result[0]["url"], str)
    assert isinstance(result[0]["date_posted"], str)


def test_format_jobs_multiple_jobs_each_has_required_keys() -> None:
    """Multiple raw jobs produce formatted jobs each with all required keys."""
    raw = [
        _fake_raw_adzuna_job(title="DevOps Engineer", company_name="Beta Inc"),
        _fake_raw_adzuna_job(title="Data Analyst", salary_min=None, salary_max=None),
    ]
    result = format_jobs(raw)
    assert len(result) == 2
    for job in result:
        for key in REQUIRED_FORMATTED_KEYS:
            assert key in job
        assert isinstance(job["salary"], str)
        assert isinstance(job["url"], str)
        assert isinstance(job["date_posted"], str)


def test_format_jobs_salary_range_format_when_min_max_present() -> None:
    """When both salary_min and salary_max present, salary string contains numbers (formatted)."""
    raw = [_fake_raw_adzuna_job(salary_min=50000, salary_max=70000)]
    result = format_jobs(raw)
    assert len(result) == 1
    salary = result[0]["salary"]
    assert "50" in salary or "50,000" in salary or "50000" in salary
    assert "70" in salary or "70,000" in salary or "70000" in salary
