"""Basic tests for resume parser helpers."""

from __future__ import annotations

import unittest

from Modules import resume_parser


class TestResumeParser(unittest.TestCase):
    """Validate non-network parser behavior."""

    def test_clean_text_normalizes_spacing(self) -> None:
        raw = "Name\r\n\r\n  Skills:\tPython   SQL\u00a0"
        cleaned = resume_parser.clean_text(raw)
        self.assertIn("Name", cleaned)
        self.assertIn("Skills: Python SQL", cleaned)
        self.assertNotIn("\r", cleaned)

    def test_extract_json_object_from_fenced_output(self) -> None:
        text = """```json
{"name": "Sai", "skills": ["Python"]}
```"""
        parsed = resume_parser._extract_json_object(text)
        self.assertEqual(parsed.get("name"), "Sai")
        self.assertEqual(parsed.get("skills"), ["Python"])

    def test_normalize_list_handles_string_input(self) -> None:
        result = resume_parser._normalize_list("Python, SQL; AWS|ML")
        self.assertEqual(result, ["Python", "SQL", "AWS", "ML"])


if __name__ == "__main__":
    unittest.main()

