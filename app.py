"""CareerMatch AI Streamlit application."""

from __future__ import annotations

import json
import os
import shutil
import uuid
import logging
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from Modules.job_search import build_search_query, format_jobs, search_jobs_detailed
from Modules.matcher import (
    deep_analysis,
    embed_and_store_jobs,
    embed_and_store_resume,
    get_top_n_jobs,
    rank_jobs,
)
from Modules.resume_builder import build_tailored_resume, generate_docx
from Modules.resume_parser import parse_resume


def _load_environment() -> None:
    """Load environment variables from .config first, then fallback to .env."""
    try:
        project_root = os.path.dirname(__file__)
        config_path = os.path.join(project_root, ".config")
        env_path = os.path.join(project_root, ".env")
        if os.path.isfile(config_path):
            load_dotenv(dotenv_path=config_path, override=True)
        else:
            load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        load_dotenv()


_load_environment()

APP_TITLE = "CareerMatch AI"
TEMP_DIR = "temp_uploads"
DOCX_DIR = "generated_resumes"
CHROMA_DIR = "chroma_db"

# Load project configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CONFIG_PATH, encoding="utf-8") as _cfg_file:
        _CONFIG = json.load(_cfg_file)
except Exception:
    _CONFIG = {}

JOB_SEARCH_CONFIG = _CONFIG.get("job_search", {}) if isinstance(_CONFIG.get("job_search", {}), dict) else {}

# Country markets (can be edited in config.json under job_search.markets)
ADZUNA_COUNTRY_OPTIONS = JOB_SEARCH_CONFIG.get("markets", {}) or {
    "United States": "us",
    "United Kingdom": "gb",
    "India": "in",
    "Canada": "ca",
    "Australia": "au",
    "Germany": "de",
    "France": "fr",
    "Italy": "it",
    "Netherlands": "nl",
    "Poland": "pl",
    "Singapore": "sg",
    "New Zealand": "nz",
    "Belgium": "be",
    "Brazil": "br",
    "South Africa": "za",
}

# Fixed search location for fetching jobs; users can still filter by actual job locations later.
DEFAULT_SEARCH_LOCATION = "Worldwide"

# Minimum match score (0–100) required for jobs to be kept in ranked lists.
# Configurable in config.json under job_search.min_match_score.
MIN_MATCH_SCORE = float(JOB_SEARCH_CONFIG.get("min_match_score", 0) or 0)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    try:
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            reset_chroma_for_new_session()

        defaults = {
            "resume_data": None,
            "jobs_list": [],
            "ranked_jobs": [],
            "top_10_jobs": [],
            "top_3_jobs": [],
            "deep_analyses": [],
            "tailored_resumes": [],
            "docx_paths": [],
            "search_country_label": "United States",
            "available_locations": [],
            "selected_job_location": "All Locations",
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    except Exception as exc:
        logger.exception("Failed to initialize session state")
        st.error(f"Failed to initialize session state: {exc}")


def reset_chroma_for_new_session() -> None:
    """Reset local ChromaDB folder for each new user session."""
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
    except Exception as exc:
        logger.warning("Could not reset ChromaDB storage", exc_info=exc)
        st.warning(f"Could not reset ChromaDB storage: {exc}")


def clear_downstream_state() -> None:
    """Clear downstream processing state after new upload."""
    try:
        st.session_state.jobs_list = []
        st.session_state.ranked_jobs = []
        st.session_state.top_10_jobs = []
        st.session_state.top_3_jobs = []
        st.session_state.deep_analyses = []
        st.session_state.tailored_resumes = []
        st.session_state.docx_paths = []
    except Exception as exc:
        logger.exception("Failed to clear downstream state")
        st.error(f"Failed to clear downstream state: {exc}")


def render_tags(items: List[str]) -> None:
    """Render list items as tag-like chips using markdown."""
    try:
        if not items:
            st.caption("No items available.")
            return
        tags = " ".join([f"`{str(item).strip()}`" for item in items if str(item).strip()])
        st.markdown(tags)
    except Exception as exc:
        logger.exception("Failed to render tags")
        st.error(f"Failed to render tags: {exc}")


def score_label(score: float) -> str:
    """Return color coded score label."""
    try:
        if score >= 70:
            return "green"
        if score >= 45:
            return "orange"
        return "red"
    except Exception:
        return "gray"


def process_resume_upload(uploaded_file: Any) -> None:
    """Process uploaded PDF and update session state."""
    try:
        if not uploaded_file:
            st.warning("Please choose a resume PDF before parsing.")
            return
        os.makedirs(TEMP_DIR, exist_ok=True)
        file_name = f"{st.session_state.session_id}_resume.pdf"
        temp_pdf_path = os.path.join(TEMP_DIR, file_name)
        with open(temp_pdf_path, "wb") as pdf_file:
            pdf_file.write(uploaded_file.getbuffer())

        with st.spinner("Parsing resume with Gemini..."):
            parsed = parse_resume(temp_pdf_path)
        if not parsed.get("raw_text", "").strip():
            st.warning(
                "We could not extract text from this PDF. Please upload a text-based resume PDF."
            )
            return
        st.session_state.resume_data = parsed
        clear_downstream_state()
        st.success("Resume parsed successfully.")
    except Exception as exc:
        logger.exception("Failed to process resume upload")
        st.error(
            "We couldn't process the uploaded resume right now. "
            "Please try another PDF or retry in a moment."
        )


def run_job_pipeline() -> None:
    """Search, embed, rank, and select top jobs."""
    try:
        resume_data = st.session_state.resume_data
        if not resume_data:
            st.warning("Upload and parse a resume first.")
            return

        structured = resume_data.get("structured_info", {})
        keywords = resume_data.get("keywords", [])
        query = build_search_query(keywords, structured)
        location = DEFAULT_SEARCH_LOCATION
        selected_country_label = str(st.session_state.search_country_label).strip()
        country_code = ADZUNA_COUNTRY_OPTIONS.get(selected_country_label, "us")

        with st.spinner("Fetching 100 jobs from Adzuna..."):
            search_result = search_jobs_detailed(
                query=query,
                location=location,
                num_jobs=100,
                country=country_code,
            )
            raw_jobs = search_result.get("jobs", [])
            jobs = format_jobs(raw_jobs)

        if not jobs:
            _show_adzuna_failure_message(search_result)
            return

        with st.spinner("Embedding and ranking jobs..."):
            embed_and_store_resume(resume_data.get("raw_text", ""))
            embed_and_store_jobs(jobs)
            ranked = rank_jobs(resume_data, jobs)

        # Apply minimum match score filtering, if configured.
        filtered_ranked = ranked
        if MIN_MATCH_SCORE > 0:
            strong_matches = [
                job for job in ranked
                if float(job.get("score", 0.0)) >= MIN_MATCH_SCORE
            ]
            if strong_matches:
                if len(strong_matches) < len(ranked):
                    st.info(
                        f"Only {len(strong_matches)} of {len(ranked)} jobs met the minimum match score "
                        f"of {MIN_MATCH_SCORE:.0f}%. Showing only strong matches."
                    )
                filtered_ranked = strong_matches
            else:
                filtered_ranked = []
                st.info(
                    f"No jobs met the minimum match score of {MIN_MATCH_SCORE:.0f}%. "
                    "Please improve your resume/keywords or lower the threshold in config.json."
                )

        st.session_state.jobs_list = jobs
        st.session_state.ranked_jobs = filtered_ranked
        unique_locations = sorted(
            {
                str(job.get("location", "")).strip()
                for job in filtered_ranked
                if str(job.get("location", "")).strip()
            }
        )
        st.session_state.available_locations = ["All Locations"] + unique_locations
        st.session_state.selected_job_location = "All Locations"
        st.session_state.top_10_jobs = get_top_n_jobs(filtered_ranked, n=10)
        st.session_state.top_3_jobs = get_top_n_jobs(filtered_ranked, n=3)
        st.session_state.deep_analyses = []
        st.session_state.tailored_resumes = []
        st.session_state.docx_paths = []
        st.success(f"Job matching complete. Retrieved {len(jobs)} jobs.")
        if search_result.get("query_fallback_used"):
            st.info(
                "No jobs were found for your resume keywords in this market; results below use diverse role searches (e.g. software developer, data analyst, cloud engineer, DevOps). Ranking and tailored resumes still use your resume."
            )
        elif search_result.get("fallback_used"):
            st.info(
                "No jobs were found for the selected location; results below are from a worldwide search."
            )
    except Exception as exc:
        logger.exception("Failed to run job pipeline")
        st.error(
            "Job matching failed. Check API keys/network and try again."
        )


def _show_adzuna_failure_message(search_result: Dict[str, Any]) -> None:
    """Show user-friendly Adzuna error details from search result."""
    error_code = str(search_result.get("error_code", "")).strip()
    if error_code == "missing_credentials":
        st.error(
            "Adzuna credentials are missing. Please set ADZUNA_APP_ID and "
            "ADZUNA_APP_KEY in `.env`, then restart the app."
        )
    elif error_code == "request_failed":
        st.error(
            "Adzuna API request failed for the selected market. "
            "Check internet, API key validity, or quota limits and retry."
        )
    elif error_code == "empty_results":
        st.info(
            "No jobs matched this query/location in the selected country market. "
            "Try another location or country."
        )
    else:
        st.warning("No jobs found for this run. Please retry with different filters.")


def run_adzuna_connection_check() -> None:
    """Run a quick Adzuna connectivity and credentials check."""
    try:
        selected_country_label = str(st.session_state.search_country_label).strip()
        country_code = ADZUNA_COUNTRY_OPTIONS.get(selected_country_label, "us")
        with st.spinner("Checking Adzuna connection..."):
            check_result = search_jobs_detailed(
                query="software engineer",
                location="Worldwide",
                num_jobs=1,
                country=country_code,
            )
            jobs = check_result.get("jobs", [])

        if jobs:
            st.success(
                f"Adzuna connection is active for {selected_country_label}. "
                "Credentials and API access look good."
            )
            return

        _show_adzuna_failure_message(check_result)
    except Exception:
        logger.exception("Failed Adzuna connection check")
        st.error("Unable to complete Adzuna connection check right now.")


def apply_location_filter(selected_location: str) -> None:
    """Filter ranked jobs by selected location and refresh top lists."""
    try:
        ranked_jobs = st.session_state.ranked_jobs or []
        if not ranked_jobs:
            st.session_state.top_10_jobs = []
            st.session_state.top_3_jobs = []
            return

        normalized_selected = str(selected_location).strip().lower()
        if not normalized_selected or normalized_selected == "all locations":
            filtered = ranked_jobs
        else:
            filtered = [
                job
                for job in ranked_jobs
                if str(job.get("location", "")).strip().lower() == normalized_selected
            ]

        st.session_state.top_10_jobs = get_top_n_jobs(filtered, n=10)
        st.session_state.top_3_jobs = get_top_n_jobs(filtered, n=3)
    except Exception:
        logger.exception("Failed applying location filter")
        st.warning("Could not filter by location. Showing existing results.")


def run_deep_analysis() -> None:
    """Run deep analysis for top 3 jobs."""
    try:
        resume_data = st.session_state.resume_data
        top_3_jobs = st.session_state.top_3_jobs
        if not resume_data or not top_3_jobs:
            st.warning("Top jobs are not ready.")
            return

        analyses = []
        with st.spinner("Running deep analysis for top 3 jobs..."):
            for job in top_3_jobs:
                analyses.append(deep_analysis(resume_data, job))
        st.session_state.deep_analyses = analyses
        st.success("Deep analysis generated successfully.")
    except Exception as exc:
        logger.exception("Failed to run deep analysis")
        st.error("Deep analysis failed. Please retry after job matching.")


def generate_tailored_resumes() -> None:
    """Generate tailored resumes and DOCX files for top 3 jobs."""
    try:
        resume_data = st.session_state.resume_data
        top_3_jobs = st.session_state.top_3_jobs
        if not resume_data or not top_3_jobs:
            st.warning("Top 3 jobs are required before generating resumes.")
            return

        os.makedirs(DOCX_DIR, exist_ok=True)
        tailored_resumes: List[Dict[str, Any]] = []
        docx_paths: List[str] = []

        with st.spinner("Building tailored resumes and DOCX files..."):
            for idx, job in enumerate(top_3_jobs, start=1):
                tailored = build_tailored_resume(resume_data, job)
                output_name = f"{st.session_state.session_id}_tailored_resume_{idx}.docx"
                output_path = os.path.join(DOCX_DIR, output_name)
                saved_path = generate_docx(tailored, output_path)
                tailored_resumes.append(tailored)
                docx_paths.append(saved_path)

        st.session_state.tailored_resumes = tailored_resumes
        st.session_state.docx_paths = docx_paths
        st.success("Tailored resumes generated successfully.")
    except Exception as exc:
        logger.exception("Failed to generate tailored resumes")
        st.error(
            "Failed to generate tailored resumes. Please retry or rerun matching."
        )


def page_home() -> None:
    """Render home page with resume upload."""
    try:
        st.header("Page 1 - Home")
        st.write("Upload your resume PDF to begin CareerMatch AI analysis.")
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Parse Resume"):
                process_resume_upload(uploaded_file)
    except Exception as exc:
        logger.exception("Home page rendering error")
        st.error(f"Home page error: {exc}")


def page_resume_analysis() -> None:
    """Render resume analysis page."""
    try:
        st.header("Page 2 - Resume Analysis")
        resume_data = st.session_state.resume_data
        if not resume_data:
            st.info("Upload and parse your resume on Home page first.")
            return

        structured = resume_data.get("structured_info", {})
        st.subheader("Candidate Information")
        st.write(f"**Name:** {structured.get('name', '') or 'N/A'}")
        st.write(f"**Email:** {structured.get('email', '') or 'N/A'}")

        st.subheader("Skills")
        render_tags(structured.get("skills", []))

        st.subheader("Top ATS Keywords")
        render_tags(resume_data.get("keywords", []))
    except Exception as exc:
        logger.exception("Resume analysis page rendering error")
        st.error(f"Resume analysis page error: {exc}")


def page_job_matches() -> None:
    """Render top 10 matched jobs page."""
    try:
        st.header("Page 3 - Job Matches")
        country_labels = list(ADZUNA_COUNTRY_OPTIONS.keys())
        current_country_label = str(st.session_state.search_country_label).strip()
        if current_country_label not in country_labels:
            current_country_label = country_labels[0]
        st.session_state.search_country_label = st.selectbox(
            "Job Market Country",
            options=country_labels,
            index=country_labels.index(current_country_label),
            help="Choose the Adzuna market to search jobs from.",
        )

        st.caption(
            "Adzuna credentials must be set in `.env` as ADZUNA_APP_ID and ADZUNA_APP_KEY. "
            "Jobs are initially fetched with a worldwide search; you can still filter by job location below."
        )

        action_col_1, action_col_2 = st.columns(2)
        if action_col_1.button("Search and Rank 100 Jobs"):
            run_job_pipeline()
        if action_col_2.button("Check Adzuna Connection"):
            run_adzuna_connection_check()

        if not st.session_state.ranked_jobs:
            st.info("Run job search to view top 10 matches.")
            return

        available_locations = st.session_state.available_locations or ["All Locations"]
        if st.session_state.selected_job_location not in available_locations:
            st.session_state.selected_job_location = "All Locations"

        selected_location = st.selectbox(
            "Filter by Available Job Location",
            options=available_locations,
            index=available_locations.index(st.session_state.selected_job_location),
            help="Locations in this list are derived from the currently fetched 100 jobs.",
        )
        if selected_location != st.session_state.selected_job_location:
            st.session_state.selected_job_location = selected_location
        apply_location_filter(st.session_state.selected_job_location)

        top_10 = st.session_state.top_10_jobs
        if not top_10:
            st.info("No jobs matched the selected location filter.")
            return

        st.subheader("Top 10 Ranked Jobs")
        for idx, job in enumerate(top_10, start=1):
            score = float(job.get("score", 0.0))
            color = score_label(score)
            st.markdown(f"### {idx}. {job.get('title', 'Untitled')} - {job.get('company', 'Unknown')}")
            st.write(f"Location: {job.get('location', 'N/A')}")
            st.write(f"Salary: {job.get('salary', 'Not disclosed')}")
            st.write(f"Apply: {job.get('url', 'N/A')}")
            st.markdown(f"**Match Score:** :{color}[{score:.2f}%]")
            st.progress(min(max(score / 100.0, 0.0), 1.0))
            st.write(f"Matched Skills: {', '.join(job.get('matched_skills', [])[:12]) or 'None'}")
            st.write(f"Missing Skills: {', '.join(job.get('missing_skills', [])[:12]) or 'None'}")
            st.divider()

        # All ranked jobs table (same location filter as top 10)
        ranked_jobs = st.session_state.ranked_jobs or []
        norm_loc = str(st.session_state.selected_job_location or "").strip().lower()
        if norm_loc and norm_loc != "all locations":
            filtered_ranked = [
                j for j in ranked_jobs
                if str(j.get("location", "")).strip().lower() == norm_loc
            ]
        else:
            filtered_ranked = ranked_jobs

        if filtered_ranked:
            st.subheader("All Ranked Jobs")
            st.caption(f"Showing all {len(filtered_ranked)} jobs (same location filter as above).")
            table_rows = []
            for idx, job in enumerate(filtered_ranked, start=1):
                table_rows.append({
                    "Rank": idx,
                    "Title": str(job.get("title", "")).strip() or "—",
                    "Company": str(job.get("company", "")).strip() or "—",
                    "Location": str(job.get("location", "")).strip() or "—",
                    "Score (%)": round(float(job.get("score", 0.0)), 2),
                    "Salary": str(job.get("salary", "Not disclosed")).strip() or "—",
                    "Apply": str(job.get("url", "")).strip() or "",
                    "Matched Skills": ", ".join((job.get("matched_skills") or [])[:8]) or "—",
                })
            st.dataframe(
                table_rows,
                column_config={
                    "Apply": st.column_config.LinkColumn("Apply", display_text="Open"),
                },
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        logger.exception("Job matches page rendering error")
        st.error(f"Job matches page error: {exc}")


def page_deep_analysis() -> None:
    """Render deep analysis for top 3 jobs."""
    try:
        st.header("Page 4 - Deep Analysis")
        top_3 = st.session_state.top_3_jobs
        if not top_3:
            st.info("Top 3 jobs will appear after running job matching.")
            return

        if st.button("Generate Deep Analysis for Top 3"):
            run_deep_analysis()

        analyses = st.session_state.deep_analyses
        if not analyses:
            st.info("Generate deep analysis to view insights.")
            return

        for idx, (job, analysis) in enumerate(zip(top_3, analyses), start=1):
            st.markdown(f"### {idx}. {job.get('title', 'Untitled')} - {job.get('company', 'Unknown')}")
            st.write(f"Overall Score: {analysis.get('overall_score', 0)}%")
            st.write(f"Experience Fit: {analysis.get('experience_fit', 'N/A')}")
            st.write(f"Education Fit: {analysis.get('education_fit', 'N/A')}")
            st.write(f"Matched Skills: {', '.join(analysis.get('matched_skills', [])[:15]) or 'None'}")
            st.write(f"Missing Skills: {', '.join(analysis.get('missing_skills', [])[:15]) or 'None'}")
            st.write("Recommendations:")
            for rec in analysis.get("recommendations", []):
                st.write(f"- {rec}")
            st.divider()
    except Exception as exc:
        logger.exception("Deep analysis page rendering error")
        st.error(f"Deep analysis page error: {exc}")


def page_tailored_resumes() -> None:
    """Render tailored resume generation and downloads."""
    try:
        st.header("Page 5 - Tailored Resumes")
        top_3 = st.session_state.top_3_jobs
        if not top_3:
            st.info("Top 3 jobs are required before generating tailored resumes.")
            return

        if st.button("Generate 3 Tailored ATS Resumes"):
            generate_tailored_resumes()

        resumes = st.session_state.tailored_resumes
        docx_paths = st.session_state.docx_paths
        if not resumes or not docx_paths:
            st.info("Generate tailored resumes to enable downloads.")
            return

        for idx, (resume, path, job) in enumerate(zip(resumes, docx_paths, top_3), start=1):
            st.markdown(f"### Tailored Resume {idx}")
            st.write(f"Target Job: {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
            st.write(f"Matched Skills Used: {', '.join(resume.get('matched_skills', [])[:12]) or 'None'}")
            st.write(f"Prioritized Skills: {', '.join(resume.get('skills', [])[:12]) or 'None'}")
            if path and os.path.exists(path):
                with open(path, "rb") as file_data:
                    st.download_button(
                        label=f"Download Resume {idx} (DOCX)",
                        data=file_data.read(),
                        file_name=os.path.basename(path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_resume_{idx}",
                    )
            else:
                st.warning("DOCX file not found.")
            st.divider()
    except Exception as exc:
        logger.exception("Tailored resumes page rendering error")
        st.error(f"Tailored resumes page error: {exc}")


def main() -> None:
    """Application entrypoint."""
    try:
        st.set_page_config(page_title=APP_TITLE, page_icon=":briefcase:", layout="wide")
        init_session_state()
        st.title(APP_TITLE)
        st.caption("RAG-powered resume analysis, job matching, and ATS resume generation.")

        page_options = [
            "HOME",
            "RESUME ANALYSIS",
            "JOB MATCHES",
            "DEEP ANALYSIS",
            "TAILORED RESUMES",
        ]
        selected_page = st.sidebar.radio("Navigate", page_options)

        if selected_page == "HOME":
            page_home()
        elif selected_page == "RESUME ANALYSIS":
            page_resume_analysis()
        elif selected_page == "JOB MATCHES":
            page_job_matches()
        elif selected_page == "DEEP ANALYSIS":
            page_deep_analysis()
        elif selected_page == "TAILORED RESUMES":
            page_tailored_resumes()
    except Exception as exc:
        logger.exception("Application error")
        st.error(f"Application error: {exc}")


if __name__ == "__main__":
    main()
