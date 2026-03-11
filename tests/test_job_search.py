"""Basic tests for job search formatting and query building."""

from __future__ import annotations

import unittest

from Modules.job_search import build_search_query, format_jobs


class TestJobSearch(unittest.TestCase):
    """Validate deterministic job search utility behavior."""

    def test_build_search_query_deduplicates_terms(self) -> None:
        query = build_search_query(
            keywords=["Python", "AWS", "python"],
            structured_info={"skills": ["SQL", "AWS"]},
        )
        self.assertIn("Python", query)
        self.assertIn("AWS", query)
        self.assertIn("SQL", query)

    def test_format_jobs_handles_optional_fields(self) -> None:
        raw_jobs = [
            {
                "title": "Backend Engineer",
                "company": {"display_name": "Acme"},
                "location": {"display_name": "Bengaluru"},
                "description": "Python and SQL",
                "salary_min": 1000000,
                "salary_max": 2000000,
                "redirect_url": "https://example.com/job/1",
                "created": "2026-03-06",
            },
            {
                "title": "Data Analyst",
                "company": {},
                "location": {},
                "description": "",
            },
        ]
        formatted = format_jobs(raw_jobs)
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["company"], "Acme")
        self.assertEqual(formatted[0]["salary"], "1,000,000 - 2,000,000")
        self.assertEqual(formatted[1]["company"], "Unknown Company")
        self.assertEqual(formatted[1]["salary"], "Not disclosed")


if __name__ == "__main__":
    unittest.main()

