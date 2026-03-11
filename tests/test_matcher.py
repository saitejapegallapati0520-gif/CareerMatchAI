"""Basic tests for ranking utilities."""

from __future__ import annotations

import unittest

from Modules import matcher


def _fake_embed_text(text: str) -> list[float]:
    """Deterministic embedding stub for tests."""
    lowered = text.lower()
    return [
        1.0 if "python" in lowered else 0.0,
        1.0 if "aws" in lowered else 0.0,
        1.0 if "sql" in lowered else 0.0,
    ]


class TestMatcher(unittest.TestCase):
    """Validate ranking behavior without external APIs."""

    def test_rank_jobs_sorts_best_match_first(self) -> None:
        original_embed = matcher._embed_text
        matcher._embed_text = _fake_embed_text
        try:
            resume_data = {
                "raw_text": "Python developer with AWS and SQL experience",
                "structured_info": {"skills": ["Python", "AWS", "SQL"]},
                "keywords": ["python", "aws", "sql"],
            }
            jobs_list = [
                {
                    "title": "Python Backend Engineer",
                    "description": "Need Python AWS SQL",
                    "company": "GoodFit",
                },
                {
                    "title": "Graphic Designer",
                    "description": "Need Photoshop Illustrator",
                    "company": "LowFit",
                },
            ]
            ranked = matcher.rank_jobs(resume_data, jobs_list)
            self.assertEqual(len(ranked), 2)
            self.assertEqual(ranked[0]["company"], "GoodFit")
            self.assertGreaterEqual(ranked[0]["score"], ranked[1]["score"])
            self.assertIn("matched_skills", ranked[0])
            self.assertIn("vector_score", ranked[0])
        finally:
            matcher._embed_text = original_embed


if __name__ == "__main__":
    unittest.main()

