"""Resume parsing utilities for CareerMatch AI."""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Dict, List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
logger = logging.getLogger(__name__)


def _invoke_with_fallback(prompt: str) -> str:
    """Invoke Gemini with model fallback and return response text."""
    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment.")
        model_candidates = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
        last_error: Exception | None = None
        for model_name in model_candidates:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.1,
                )
                response = llm.invoke(prompt)
                return str(getattr(response, "content", "") or "")
            except Exception as model_exc:
                last_error = model_exc
                continue
        raise RuntimeError(str(last_error) if last_error else "Unknown LLM invocation error.")
    except Exception as exc:
        raise RuntimeError(f"Failed to invoke Gemini model: {exc}") from exc


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from text."""
    try:
        if not text:
            return {}
        cleaned = text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {}
        return json.loads(match.group(0))
    except Exception:
        return {}


def _normalize_list(value: Any) -> List[str]:
    """Convert any value into a clean list of non-empty strings."""
    try:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[,\n;|]", value) if item.strip()]
        return []
    except Exception:
        return []


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file path."""
    try:
        logger.info("Extracting text from PDF")
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages: List[str] = []
        for page in doc:
            pages.append(page.get_text("text") or "")
        doc.close()

        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("PDF contains no extractable text.")
        return text
    except Exception as exc:
        logger.warning("Error in extract_text_from_pdf: %s", exc)
        return ""


def clean_text(text: str) -> str:
    """Clean and normalize extracted resume text."""
    try:
        logger.info("Cleaning extracted text")
        if not text:
            return ""
        cleaned = re.sub(r"\r", "\n", text)
        cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\u00a0", " ", cleaned)
        cleaned = cleaned.strip()
        return cleaned
    except Exception as exc:
        logger.warning("Error in clean_text: %s", exc)
        return ""


def extract_resume_info(raw_text: str) -> Dict[str, Any]:
    """Extract structured resume information using Gemini."""
    try:
        logger.info("Extracting structured resume information")
        if not raw_text:
            return {
                "name": "",
                "email": "",
                "skills": [],
                "experience": [],
                "education": [],
                "projects": [],
                "certifications": [],
                "summary": "",
                "top_ats_keywords": [],
            }

        prompt = f"""
You are an expert resume parser.
Extract only information explicitly present in the resume text.
Do not fabricate details.

Return strict JSON with these exact keys:
name (string),
email (string),
skills (array of strings),
experience (array of strings),
education (array of strings),
projects (array of strings),
certifications (array of strings),
summary (string),
top_ats_keywords (array of up to 30 strings)

Resume text:
{raw_text}
"""
        content = _invoke_with_fallback(prompt)
        parsed = _extract_json_object(content)

        return {
            "name": str(parsed.get("name", "")).strip(),
            "email": str(parsed.get("email", "")).strip(),
            "skills": _normalize_list(parsed.get("skills", [])),
            "experience": _normalize_list(parsed.get("experience", [])),
            "education": _normalize_list(parsed.get("education", [])),
            "projects": _normalize_list(parsed.get("projects", [])),
            "certifications": _normalize_list(parsed.get("certifications", [])),
            "summary": str(parsed.get("summary", "")).strip(),
            "top_ats_keywords": _normalize_list(parsed.get("top_ats_keywords", []))[:30],
        }
    except Exception as exc:
        logger.warning("Error in extract_resume_info: %s", exc)
        return {
            "name": "",
            "email": "",
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "summary": "",
            "top_ats_keywords": [],
        }


def extract_keywords(raw_text: str) -> List[str]:
    """Extract top ATS-focused keywords from resume text."""
    try:
        logger.info("Extracting ATS keywords")
        if not raw_text:
            return []

        prompt = f"""
From the resume text below, extract the top 30 ATS-friendly keywords.
Only use words or phrases already present in the text.
Return strict JSON: {{"keywords": ["...", "..."]}}

Resume text:
{raw_text}
"""
        content = _invoke_with_fallback(prompt)
        parsed = _extract_json_object(content)
        keywords = _normalize_list(parsed.get("keywords", []))[:30]

        # Final cleanup for duplicates and very short noise terms.
        seen = set()
        unique_keywords: List[str] = []
        for keyword in keywords:
            key = keyword.lower()
            if key and len(key) >= 2 and key not in seen:
                unique_keywords.append(keyword)
                seen.add(key)
        return unique_keywords[:30]
    except Exception as exc:
        logger.warning("Error in extract_keywords: %s", exc)
        return []


def parse_resume(pdf_path: str) -> Dict[str, Any]:
    """Parse a resume PDF and return raw text, structured info, and keywords."""
    try:
        logger.info("Starting full resume parsing pipeline")
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        structured_info = extract_resume_info(cleaned_text)
        keywords = extract_keywords(cleaned_text)

        if not keywords and structured_info.get("top_ats_keywords"):
            keywords = _normalize_list(structured_info.get("top_ats_keywords", []))[:30]

        result = {
            "raw_text": cleaned_text,
            "structured_info": structured_info,
            "keywords": keywords,
        }
        logger.info("Resume parsing complete")
        return result
    except Exception as exc:
        logger.exception("Error in parse_resume")
        return {
            "raw_text": "",
            "structured_info": {
                "name": "",
                "email": "",
                "skills": [],
                "experience": [],
                "education": [],
                "projects": [],
                "certifications": [],
                "summary": "",
                "top_ats_keywords": [],
            },
            "keywords": [],
        }


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Running module self-test")
        test_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_resume.pdf")
        if not os.path.exists(test_pdf):
            logger.info("test_resume.pdf not found. Place a file in project root to test parsing.")
        else:
            parsed_resume = parse_resume(test_pdf)
            logger.info("Parsed name: %s", parsed_resume["structured_info"].get("name", ""))
            logger.info("Extracted keywords count: %s", len(parsed_resume.get("keywords", [])))
    except Exception as exc:
        logger.exception("Self-test failed")
