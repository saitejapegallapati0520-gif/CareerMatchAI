"""Job search module using Adzuna API."""

from __future__ import annotations

import json
import math
import os
import logging
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


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

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"

# Defaults when config.json is missing or invalid
DEFAULT_FALLBACK_QUERIES = [
    "software developer",
    "data analyst",
    "cloud engineer",
    "devops engineer",
    "data engineer",
    "backend developer",
    "software engineer",
]
DEFAULT_FALLBACK_JOBS_PER_QUERY = 20


def _load_job_search_config() -> Dict[str, Any]:
    """Load job_search section from project config.json. Returns defaults if missing."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            js = data.get("job_search")
            if isinstance(js, dict):
                return js
    except Exception as exc:
        logger.warning("Could not load config.json, using defaults: %s", exc)
    return {
        "fallback_queries": DEFAULT_FALLBACK_QUERIES,
        "fallback_jobs_per_query": DEFAULT_FALLBACK_JOBS_PER_QUERY,
    }


def build_search_query(keywords: List[str], structured_info: Dict[str, Any]) -> str:
    """Build a search query from extracted resume keywords and skills."""
    try:
        logger.info("Building search query from resume keywords")
        cleaned_keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        skills = [str(s).strip() for s in structured_info.get("skills", []) if str(s).strip()]

        merged_terms: List[str] = []
        seen = set()
        for term in cleaned_keywords + skills:
            lowered = term.lower()
            if lowered not in seen:
                merged_terms.append(term)
                seen.add(lowered)

        if not merged_terms:
            return "software engineer python"
        return " ".join(merged_terms[:12])
    except Exception as exc:
        logger.exception("Error in build_search_query")
        return "software engineer python"


def search_jobs(query: str, location: str, num_jobs: int = 100, country: str | None = None) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper for basic job fetching."""
    result = search_jobs_detailed(query=query, location=location, num_jobs=num_jobs, country=country)
    return result.get("jobs", [])


def search_jobs_detailed(
    query: str, location: str, num_jobs: int = 100, country: str | None = None
) -> Dict[str, Any]:
    """Fetch jobs from Adzuna API."""
    try:
        logger.info("Searching live jobs from Adzuna")
        app_id = os.getenv("ADZUNA_APP_ID", "").strip()
        app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
        country_code = str(country or os.getenv("ADZUNA_COUNTRY", "us")).strip().lower()

        if not app_id or not app_key:
            logger.warning("Missing ADZUNA_APP_ID/ADZUNA_APP_KEY in .env")
            return {
                "jobs": [],
                "error_code": "missing_credentials",
                "error_message": "Missing ADZUNA_APP_ID or ADZUNA_APP_KEY in .env.",
            }

        if not query.strip():
            query = "software engineer"
        normalized_location = str(location).strip()
        if normalized_location.lower() in {"worldwide", "global", "any", "anywhere"}:
            normalized_location = ""

        jobs: List[Dict[str, Any]] = []
        per_page = 50
        pages_needed = max(1, math.ceil(num_jobs / per_page))
        page_failures = 0

        for page in range(1, pages_needed + 1):
            url = f"{ADZUNA_BASE_URL}/{country_code}/search/{page}"
            params = {
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": per_page,
                "what": query,
                "content-type": "application/json",
            }
            if normalized_location:
                params["where"] = normalized_location
            try:
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results", []) or []
                jobs.extend(results)
                logger.info("Retrieved Adzuna page %s with %s jobs", page, len(results))
                if not results:
                    break
            except Exception as page_exc:
                logger.warning("Failed fetching Adzuna page %s: %s", page, page_exc)
                page_failures += 1
                continue

            if len(jobs) >= num_jobs:
                break

        trimmed_jobs = jobs[:num_jobs]
        if trimmed_jobs:
            return {"jobs": trimmed_jobs, "error_code": None, "error_message": ""}
        if page_failures >= pages_needed:
            return {
                "jobs": [],
                "error_code": "request_failed",
                "error_message": "All Adzuna API page requests failed.",
            }
        # Fallback: if a location was specified and we got 0 results, retry without location (Worldwide)
        if normalized_location:
            logger.info("Retrying search without location (Worldwide fallback)")
            jobs_fallback: List[Dict[str, Any]] = []
            for page in range(1, pages_needed + 1):
                url = f"{ADZUNA_BASE_URL}/{country_code}/search/{page}"
                params = {
                    "app_id": app_id,
                    "app_key": app_key,
                    "results_per_page": per_page,
                    "what": query,
                    "content-type": "application/json",
                }
                try:
                    response = requests.get(url, params=params, timeout=20)
                    response.raise_for_status()
                    payload = response.json()
                    results = payload.get("results", []) or []
                    jobs_fallback.extend(results)
                    logger.info("Retrieved Adzuna page %s (fallback) with %s jobs", page, len(results))
                    if not results:
                        break
                except Exception as page_exc:
                    logger.warning("Failed fetching Adzuna page %s (fallback): %s", page, page_exc)
                    continue
                if len(jobs_fallback) >= num_jobs:
                    break
            trimmed_fallback = jobs_fallback[:num_jobs]
            if trimmed_fallback:
                return {
                    "jobs": trimmed_fallback,
                    "error_code": None,
                    "error_message": "",
                    "fallback_used": True,
                }
        # Query fallback: retry with multiple role-diverse queries from config (so users see Data Analyst, Cloud Engineer, DevOps, etc.)
        config = _load_job_search_config()
        fallback_queries = config.get("fallback_queries") or DEFAULT_FALLBACK_QUERIES
        jobs_per_query = max(1, int(config.get("fallback_jobs_per_query") or DEFAULT_FALLBACK_JOBS_PER_QUERY))
        if not isinstance(fallback_queries, list):
            fallback_queries = DEFAULT_FALLBACK_QUERIES

        seen_ids: set = set()
        jobs_query_fallback: List[Dict[str, Any]] = []
        for fq in fallback_queries:
            if len(jobs_query_fallback) >= num_jobs:
                break
            fq = str(fq).strip()
            if not fq:
                continue
            pages_fq = max(1, math.ceil(jobs_per_query / per_page))
            for page in range(1, pages_fq + 1):
                if len(jobs_query_fallback) >= num_jobs:
                    break
                url = f"{ADZUNA_BASE_URL}/{country_code}/search/{page}"
                params = {
                    "app_id": app_id,
                    "app_key": app_key,
                    "results_per_page": per_page,
                    "what": fq,
                    "content-type": "application/json",
                }
                try:
                    response = requests.get(url, params=params, timeout=20)
                    response.raise_for_status()
                    payload = response.json()
                    results = payload.get("results", []) or []
                    for job in results:
                        job_id = job.get("id") or job.get("redirect_url") or (job.get("title"), job.get("company", {}).get("display_name") if isinstance(job.get("company"), dict) else "")
                        if job_id not in seen_ids:
                            seen_ids.add(job_id)
                            jobs_query_fallback.append(job)
                    logger.info("Query fallback '%s' page %s: %s jobs (total %s)", fq, page, len(results), len(jobs_query_fallback))
                    if not results:
                        break
                except Exception as page_exc:
                    logger.warning("Failed query fallback '%s' page %s: %s", fq, page, page_exc)
                    continue
        trimmed_q = jobs_query_fallback[:num_jobs]
        if trimmed_q:
            logger.info("Query fallback returned %s diverse roles", len(trimmed_q))
            return {
                "jobs": trimmed_q,
                "error_code": None,
                "error_message": "",
                "fallback_used": True,
                "query_fallback_used": True,
            }
        return {
            "jobs": [],
            "error_code": "empty_results",
            "error_message": "No jobs found for selected query/location.",
        }
    except Exception as exc:
        logger.exception("Error in search_jobs")
        return {
            "jobs": [],
            "error_code": "unexpected_error",
            "error_message": f"Unexpected Adzuna search error: {exc}",
        }


def format_jobs(raw_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize raw Adzuna jobs into a clean schema."""
    try:
        logger.info("Formatting raw jobs")
        formatted_jobs: List[Dict[str, Any]] = []
        for item in raw_jobs or []:
            company_data = item.get("company", {}) if isinstance(item.get("company", {}), dict) else {}
            location_data = item.get("location", {}) if isinstance(item.get("location", {}), dict) else {}
            salary_min = item.get("salary_min")
            salary_max = item.get("salary_max")

            if salary_min and salary_max:
                salary = f"{int(salary_min):,} - {int(salary_max):,}"
            elif salary_min:
                salary = f"From {int(salary_min):,}"
            elif salary_max:
                salary = f"Up to {int(salary_max):,}"
            else:
                salary = "Not disclosed"

            formatted_jobs.append(
                {
                    "title": str(item.get("title", "Untitled Role")).strip(),
                    "company": str(company_data.get("display_name", "Unknown Company")).strip(),
                    "location": str(location_data.get("display_name", "Unknown Location")).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "salary": salary,
                    "url": str(item.get("redirect_url", "")).strip(),
                    "date_posted": str(item.get("created", "")).strip(),
                }
            )
        return formatted_jobs
    except Exception as exc:
        logger.exception("Error in format_jobs")
        return []


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Running module self-test")
        test_query = "python developer machine learning"
        test_location = "India"
        raw = search_jobs(test_query, test_location, num_jobs=100)
        jobs = format_jobs(raw)
        logger.info("Total fetched jobs: %s", len(raw))
        logger.info("Total formatted jobs: %s", len(jobs))
        if jobs:
            logger.info("Sample job title: %s", jobs[0].get("title", ""))
    except Exception as exc:
        logger.exception("Self-test failed")
