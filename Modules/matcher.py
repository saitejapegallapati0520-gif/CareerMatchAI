"""Matching and ranking module for CareerMatch AI."""

from __future__ import annotations

import json
import math
import os
import re
import logging
from typing import Any, Dict, List, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def _load_environment() -> None:
    """Load environment variables from .config first, then fallback to .env."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, ".config")
        env_path = os.path.join(project_root, ".env")
        if os.path.isfile(config_path):
            load_dotenv(dotenv_path=config_path, override=True)
        else:
            load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        load_dotenv()


_load_environment()
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

DEFAULT_KEYWORD_WEIGHT = 0.6
DEFAULT_VECTOR_WEIGHT = 0.4


def _load_matching_config() -> Tuple[float, float]:
    """Load keyword_weight and vector_weight from config.json. Normalize to sum to 1.0."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            matching = data.get("matching")
            if isinstance(matching, dict):
                kw = float(matching.get("keyword_weight", DEFAULT_KEYWORD_WEIGHT) or 0)
                vw = float(matching.get("vector_weight", DEFAULT_VECTOR_WEIGHT) or 0)
                if kw < 0:
                    kw = DEFAULT_KEYWORD_WEIGHT
                if vw < 0:
                    vw = DEFAULT_VECTOR_WEIGHT
                total = kw + vw
                if total > 0:
                    return (kw / total, vw / total)
    except Exception as exc:
        logger.warning("Could not load matching config, using defaults: %s", exc)
    return (DEFAULT_KEYWORD_WEIGHT, DEFAULT_VECTOR_WEIGHT)
RESUME_COLLECTION = "resume_embeddings"
JOBS_COLLECTION = "job_embeddings"


def _embed_text(text: str) -> List[float]:
    """Embed text with fallback embedding models."""
    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment.")
        model_candidates = ["models/embedding-001", "models/gemini-embedding-001", "gemini-embedding-001"]
        last_error: Exception | None = None
        for model_name in model_candidates:
            try:
                emb = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=api_key,
                )
                return emb.embed_query(text)
            except Exception as model_exc:
                last_error = model_exc
                continue
        raise RuntimeError(str(last_error) if last_error else "Unknown embedding error.")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc


def _get_chroma_client() -> chromadb.PersistentClient:
    """Create persistent ChromaDB client."""
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize ChromaDB client: {exc}") from exc


def _safe_text(text: Any) -> str:
    """Return normalized string for any input value."""
    try:
        return str(text or "").strip()
    except Exception:
        return ""


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    try:
        return re.findall(r"[a-zA-Z0-9\+\#\.]+", _safe_text(text).lower())
    except Exception:
        return []


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between vectors."""
    try:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)
    except Exception:
        return 0.0


def _get_resume_text(resume_data: Dict[str, Any]) -> str:
    """Build resume text from parsed data for vector comparisons."""
    try:
        raw_text = _safe_text(resume_data.get("raw_text", ""))
        if raw_text:
            return raw_text

        info = resume_data.get("structured_info", {}) if isinstance(resume_data.get("structured_info", {}), dict) else {}
        parts = [
            _safe_text(info.get("summary", "")),
            " ".join(info.get("skills", []) if isinstance(info.get("skills", []), list) else []),
            " ".join(info.get("experience", []) if isinstance(info.get("experience", []), list) else []),
            " ".join(info.get("projects", []) if isinstance(info.get("projects", []), list) else []),
            " ".join(info.get("education", []) if isinstance(info.get("education", []), list) else []),
            " ".join(info.get("certifications", []) if isinstance(info.get("certifications", []), list) else []),
        ]
        return " ".join([p for p in parts if p]).strip()
    except Exception:
        return ""


def embed_and_store_resume(resume_text: str) -> None:
    """Embed resume text and store in ChromaDB."""
    try:
        logger.info("Embedding and storing resume")
        if not _safe_text(resume_text):
            logger.warning("Resume text is empty. Skipping embedding")
            return

        vector = _embed_text(resume_text)
        client = _get_chroma_client()

        try:
            client.delete_collection(RESUME_COLLECTION)
        except Exception:
            pass

        collection = client.get_or_create_collection(name=RESUME_COLLECTION)
        collection.add(
            ids=["resume_1"],
            documents=[resume_text],
            embeddings=[vector],
            metadatas=[{"source": "resume"}],
        )
        logger.info("Resume embedding stored successfully")
    except Exception as exc:
        logger.exception("Error in embed_and_store_resume")


def embed_and_store_jobs(jobs_list: List[Dict[str, Any]]) -> None:
    """Embed job descriptions and store in ChromaDB."""
    try:
        logger.info("Embedding and storing jobs")
        if not jobs_list:
            logger.warning("No jobs found for embedding")
            return

        client = _get_chroma_client()

        try:
            client.delete_collection(JOBS_COLLECTION)
        except Exception:
            pass

        collection = client.get_or_create_collection(name=JOBS_COLLECTION)

        ids: List[str] = []
        docs: List[str] = []
        vectors: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, job in enumerate(jobs_list):
            description = _safe_text(job.get("description", ""))
            title = _safe_text(job.get("title", ""))
            combined_text = f"{title}\n{description}".strip()
            if not combined_text:
                continue
            try:
                vector = _embed_text(combined_text)
                ids.append(f"job_{idx}")
                docs.append(combined_text)
                vectors.append(vector)
                metadatas.append({"title": title, "company": _safe_text(job.get("company", ""))})
            except Exception as embed_exc:
                logger.warning("Failed embedding job index %s: %s", idx, embed_exc)
                continue

        if ids:
            collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas)
            logger.info("Stored %s job embeddings", len(ids))
        else:
            logger.warning("No valid jobs to store")
    except Exception as exc:
        logger.exception("Error in embed_and_store_jobs")


def calculate_match_score(resume_keywords: List[str], job: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate keyword match score between resume and a job."""
    try:
        job_text = f"{_safe_text(job.get('title', ''))} {_safe_text(job.get('description', ''))}".lower()
        normalized_keywords = [str(k).strip().lower() for k in (resume_keywords or []) if str(k).strip()]
        if not normalized_keywords:
            return {"score": 0.0, "matched_skills": [], "missing_skills": []}

        matched = [kw for kw in normalized_keywords if kw in job_text]
        missing = [kw for kw in normalized_keywords if kw not in job_text]
        score = (len(matched) / max(1, len(normalized_keywords))) * 100.0
        return {
            "score": round(score, 2),
            "matched_skills": matched,
            "missing_skills": missing,
        }
    except Exception as exc:
        logger.exception("Error in calculate_match_score")
        return {"score": 0.0, "matched_skills": [], "missing_skills": []}


def rank_jobs(resume_data: Dict[str, Any], jobs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank jobs using hybrid keyword and vector similarity scoring."""
    try:
        keyword_weight, vector_weight = _load_matching_config()
        logger.info("Ranking jobs by hybrid match score (keyword_weight=%.2f, vector_weight=%.2f)", keyword_weight, vector_weight)
        if not jobs_list:
            return []

        resume_keywords = resume_data.get("keywords", []) if isinstance(resume_data.get("keywords", []), list) else []
        resume_text = _get_resume_text(resume_data)
        resume_vector = _embed_text(resume_text) if resume_text else []

        ranked: List[Dict[str, Any]] = []
        for job in jobs_list:
            keyword_result = calculate_match_score(resume_keywords, job)

            job_text = f"{_safe_text(job.get('title', ''))}\n{_safe_text(job.get('description', ''))}".strip()
            vector_similarity = 0.0
            if job_text and resume_vector:
                try:
                    job_vector = _embed_text(job_text)
                    vector_similarity = max(0.0, _cosine_similarity(resume_vector, job_vector))
                except Exception as embed_exc:
                    logger.warning(
                        "Vector scoring failed for job '%s': %s",
                        _safe_text(job.get("title", "")),
                        embed_exc,
                    )
                    vector_similarity = 0.0

            vector_score = vector_similarity * 100.0
            final_score = (keyword_weight * keyword_result["score"]) + (vector_weight * vector_score)
            enriched_job = dict(job)
            enriched_job["score"] = round(final_score, 2)
            enriched_job["keyword_score"] = round(keyword_result["score"], 2)
            enriched_job["vector_score"] = round(vector_score, 2)
            enriched_job["matched_skills"] = keyword_result["matched_skills"]
            enriched_job["missing_skills"] = keyword_result["missing_skills"]
            ranked.append(enriched_job)

        ranked.sort(key=lambda item: item.get("score", 0), reverse=True)
        logger.info("Ranked %s jobs", len(ranked))
        return ranked
    except Exception as exc:
        logger.exception("Error in rank_jobs")
        return []


def get_top_n_jobs(ranked_jobs: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """Return top N ranked jobs."""
    try:
        logger.info("Selecting top %s jobs", n)
        if n <= 0:
            return []
        return (ranked_jobs or [])[:n]
    except Exception as exc:
        logger.exception("Error in get_top_n_jobs")
        return []


def _fit_flags(resume_data: Dict[str, Any], job: Dict[str, Any]) -> Tuple[str, str]:
    """Estimate experience and education fit for deep analysis."""
    try:
        job_text = f"{_safe_text(job.get('title', ''))} {_safe_text(job.get('description', ''))}".lower()
        experience_list = resume_data.get("structured_info", {}).get("experience", [])
        education_list = resume_data.get("structured_info", {}).get("education", [])
        exp_text = " ".join(experience_list).lower() if isinstance(experience_list, list) else ""
        edu_text = " ".join(education_list).lower() if isinstance(education_list, list) else ""

        experience_fit = "Medium"
        education_fit = "Medium"

        exp_hits = 0
        for token in ["year", "years", "engineer", "developer", "lead", "analyst"]:
            if token in exp_text and token in job_text:
                exp_hits += 1
        if exp_hits >= 3:
            experience_fit = "High"
        elif exp_hits <= 1:
            experience_fit = "Low"

        edu_hits = 0
        for token in ["bachelor", "master", "b.tech", "m.tech", "computer", "engineering"]:
            if token in edu_text and token in job_text:
                edu_hits += 1
        if edu_hits >= 2:
            education_fit = "High"
        elif edu_hits == 0:
            education_fit = "Low"

        return experience_fit, education_fit
    except Exception:
        return "Medium", "Medium"


def deep_analysis(resume_data: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """Provide deep analysis for one selected job."""
    try:
        logger.info("Running deep analysis for '%s'", _safe_text(job.get("title", "")))
        keywords = resume_data.get("keywords", []) if isinstance(resume_data.get("keywords", []), list) else []
        base = calculate_match_score(keywords, job)
        experience_fit, education_fit = _fit_flags(resume_data, job)

        recommendations: List[str] = []
        if base["missing_skills"]:
            recommendations.append(
                "Include measurable evidence for missing keywords already present in your genuine experience."
            )
            recommendations.append(
                f"Prioritize these missing skills in resume wording: {', '.join(base['missing_skills'][:8])}."
            )
        else:
            recommendations.append("Your keyword coverage is strong; focus on quantified impact in bullet points.")

        if experience_fit == "Low":
            recommendations.append("Emphasize relevant project outcomes to strengthen experience alignment.")
        if education_fit == "Low":
            recommendations.append("Highlight coursework/certifications that map directly to this role.")

        overall_score = base["score"]
        if experience_fit == "High":
            overall_score += 5
        elif experience_fit == "Low":
            overall_score -= 5
        if education_fit == "High":
            overall_score += 3
        elif education_fit == "Low":
            overall_score -= 3
        overall_score = max(0.0, min(100.0, overall_score))

        return {
            "overall_score": round(overall_score, 2),
            "matched_skills": base["matched_skills"],
            "missing_skills": base["missing_skills"],
            "experience_fit": experience_fit,
            "education_fit": education_fit,
            "recommendations": recommendations,
        }
    except Exception as exc:
        logger.exception("Error in deep_analysis")
        return {
            "overall_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "experience_fit": "Unknown",
            "education_fit": "Unknown",
            "recommendations": ["Unable to complete deep analysis due to runtime error."],
        }


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Running module self-test")
        sample_resume = {
            "raw_text": "Python developer with experience in AWS, SQL, and machine learning.",
            "structured_info": {
                "skills": ["Python", "AWS", "SQL", "Machine Learning"],
                "experience": ["3 years software engineer", "Built ML pipelines"],
                "education": ["Bachelor of Technology in Computer Science"],
            },
            "keywords": ["python", "aws", "sql", "machine learning", "nlp"],
        }
        sample_jobs = [
            {
                "title": "Python Backend Engineer",
                "company": "Example Corp",
                "location": "Remote",
                "description": "Looking for Python, AWS, SQL and API development skills.",
                "salary": "Not disclosed",
                "url": "https://example.com/job/1",
                "date_posted": "2026-03-01",
            },
            {
                "title": "Data Analyst",
                "company": "DataWorks",
                "location": "Bengaluru",
                "description": "Excel, dashboarding, and reporting experience required.",
                "salary": "Not disclosed",
                "url": "https://example.com/job/2",
                "date_posted": "2026-03-02",
            },
        ]
        embed_and_store_resume(sample_resume["raw_text"])
        embed_and_store_jobs(sample_jobs)
        ranked_jobs = rank_jobs(sample_resume, sample_jobs)
        logger.info("Ranked jobs count: %s", len(ranked_jobs))
        if ranked_jobs:
            analysis = deep_analysis(sample_resume, ranked_jobs[0])
            logger.info("Top score: %s", ranked_jobs[0].get("score", 0))
            logger.info("Deep analysis score: %s", analysis.get("overall_score", 0))
    except Exception as exc:
        logger.exception("Self-test failed")
