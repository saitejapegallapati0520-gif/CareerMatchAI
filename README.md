# CareerMatch_AI_Project
# CareerMatch AI

**RAG-powered resume analysis, job matching, and ATS-optimized resume generation.**

CareerMatch AI is a web application that parses your resume (PDF), extracts skills and ATS keywords using Google Gemini, fetches live jobs from the Adzuna API, ranks them by keyword and semantic similarity, and generates up to three tailored DOCX resumes for the top matches—without fabricating content.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Phase 1 Summary](#phase-1-summary)
- [License](#license)

---

## Overview

1. **Upload** a resume PDF on the Home page.
2. **Parse** — the system extracts structured info (name, email, skills, experience, education, projects, certifications) and top ATS keywords via Gemini.
3. **Search** — up to 100 live jobs are fetched from Adzuna for your chosen country/market.
4. **Rank** — jobs are scored using hybrid keyword match + vector similarity (ChromaDB + Gemini embeddings) and filtered by a configurable minimum match score.
5. **Top 10** jobs are shown with match scores, matched/missing skills, and apply links.
6. **Deep analysis** is available for the top 3 jobs (experience fit, education fit, recommendations).
7. **Tailored resumes** — three ATS-optimized DOCX files are generated (rephrased summary, reordered bullets, prioritized skills) and can be downloaded.

---

## Features

- PDF resume parsing with structured extraction and ATS keyword detection
- Live job search (Adzuna) with country/market selection and worldwide fallback
- Hybrid ranking: keyword overlap + semantic similarity (embeddings)
- Location filter on ranked results; optional minimum match score
- Deep analysis for top 3 jobs (matched/missing skills, fit, recommendations)
- Three tailored DOCX resumes (no fabrication; rephrase and reorder only)
- Centralized configuration via `config.json` and `.env`
- Logging across pipeline stages; user-friendly error messages

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language / runtime** | Python 3.13 |
| **Web UI** | Streamlit (single-page app, 5 logical pages: Home, Resume Analysis, Job Matches, Deep Analysis, Tailored Resumes) |
| **LLM / embeddings** | LangChain, LangChain Google GenAI, Google Generative AI; Gemini for parsing & tailoring; Gemini embeddings with model fallbacks |
| **RAG / vector store** | ChromaDB (persistent `chroma_db/`); resume + job embeddings |
| **Parsing & documents** | PyMuPDF (fitz) for PDF text; python-docx for ATS-friendly DOCX output |
| **Configuration** | python-dotenv (`.env`); `config.json` (matching weights, job search fallbacks, markets) |
| **External APIs** | Adzuna Jobs API (live job listings) |
| **Logging** | Python `logging`; module-level loggers; `LOG_LEVEL` in `.env` |
| **Testing** | pytest |

---

## Project Structure

```
CareerMatchAI/
├── app.py                 # Streamlit entrypoint; 5-page flow and session state
├── config.json            # Matching weights, job_search fallbacks, markets
├── requirements.txt       # Pinned dependencies + pytest
├── .env.example           # Template for required env vars (no secrets)
├── .gitignore
├── Modules/
│   ├── resume_parser.py   # PDF extraction, Gemini parsing, keyword extraction
│   ├── job_search.py      # Adzuna client, build_search_query, format_jobs
│   ├── matcher.py         # Embeddings, ChromaDB, rank_jobs, deep_analysis
│   └── resume_builder.py  # Tailoring (summary, experience, projects), generate_docx
├── tests/
│   ├── __init__.py
│   ├── test_resume_parser_schema.py   # Parser output schema (mocked LLM)
│   ├── test_job_search_format_jobs.py  # format_jobs keys and types
│   └── test_matcher_rank_jobs.py      # rank_jobs length, sort order (mocked embeddings)
├── chroma_db/             # Created at runtime; ChromaDB persistence (gitignored)
├── temp_uploads/          # Created at runtime; uploaded PDFs (if used)
└── generated_resumes/     # Created at runtime; output DOCX files (if used)
```

---

## Prerequisites

- **Python 3.13**
- **Google Gemini API key** — [Google AI Studio](https://aistudio.google.com/) or Google Cloud
- **Adzuna API credentials** — free at [developer.adzuna.com](https://developer.adzuna.com/) (`ADZUNA_APP_ID`, `ADZUNA_APP_KEY`)

---

## Installation

1. **Clone the repository** (or use your local project folder):

   ```bash
   git clone https://github.com/saitejapegallapati0520-gif/CareerMatchAI.git
   cd CareerMatchAI
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate      # Linux / macOS
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

### Environment variables

Copy `.env.example` to `.env` and set your keys (never commit `.env`):

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Required. Google Gemini API key for parsing and embeddings. |
| `ADZUNA_APP_ID` | Required for job search. From [developer.adzuna.com](https://developer.adzuna.com/). |
| `ADZUNA_APP_KEY` | Required for job search. |
| `ADZUNA_COUNTRY` | Default Adzuna market (e.g. `us`, `gb`, `in`). Overridable per run in the UI. |
| `LOG_LEVEL` | Optional. `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`. |

### config.json

- **`matching`** — `keyword_weight` and `vector_weight` (normalized to sum to 1.0) for hybrid scoring.
- **`job_search`** — `fallback_queries` (role keywords if primary query returns no jobs), `fallback_jobs_per_query`, `min_match_score` (0–100 filter), `markets` (country label → Adzuna country code).

---

## Usage

From the project root:

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`). Use the sidebar to move through: **Home** → **Resume Analysis** → **Job Matches** → **Deep Analysis** → **Tailored Resumes**.

---

## How It Works

- **Resume parsing** (`Modules/resume_parser.py`) — PDF text is extracted with PyMuPDF, cleaned, then sent to Gemini with a strict JSON prompt. Output: `structured_info` (name, email, skills, experience, education, projects, certifications, summary, top_ats_keywords) and a separate `keywords` list. No fabrication.

- **Job search** (`Modules/job_search.py`) — `build_search_query` builds a query from resume keywords and skills. `search_jobs_detailed` calls the Adzuna API with pagination, validates credentials, and applies fallbacks (worldwide search, then multiple role queries from config). `format_jobs` normalizes raw results to a fixed schema: `title`, `company`, `location`, `description`, `salary`, `url`, `date_posted`.

- **Matching / ranking** (`Modules/matcher.py`) — Resume and job texts are embedded with Gemini (with model fallbacks). `rank_jobs` combines keyword overlap and vector similarity using weights from `config.json`, then sorts by score. Optional `min_match_score` filtering is applied in the app. `deep_analysis` returns overall_score, matched/missing skills, experience_fit, education_fit, and recommendations.

- **Resume tailoring & DOCX** (`Modules/resume_builder.py`) — Gemini rephrases and reorders summary, experience, and projects to align with the job description without inventing content. Skills are prioritized by job relevance. `generate_docx` writes ATS-friendly DOCX with sections: Header → Summary → Skills → Experience → Projects → Education → Certifications (Calibri 11pt, no tables/images).

- **Streamlit app** (`app.py`) — Single entrypoint; all state in `st.session_state`. Handles upload, parsing, country selection, job search, Adzuna connection check, location filter, deep analysis, and tailored DOCX generation with download buttons. ChromaDB is reset per session.

---

## Testing

Tests use **pytest** and avoid real API calls (Gemini and Adzuna are mocked or bypassed).

| Test file | Coverage |
|-----------|----------|
| `tests/test_resume_parser_schema.py` | Parser output schema: `clean_text`, `extract_resume_info("")`, `extract_keywords("")`, `parse_resume` fallback keys; mocked `_invoke_with_fallback` for full schema and types. |
| `tests/test_job_search_format_jobs.py` | `format_jobs` with fake Adzuna payloads: required keys (`title`, `company`, `location`, `description`, `salary`, `url`, `date_posted`), salary formats, string types. |
| `tests/test_matcher_rank_jobs.py` | `rank_jobs` with mocked `_embed_text`: `len(ranked) == len(jobs)`, scores descending, each item has `score`, `matched_skills`, `missing_skills`; empty jobs → empty list. |

**Run all tests** from the project root:

```bash
python -m pytest tests/ -v
```

---

## Phase 1 Summary

Phase 1 focused on **stability, clarity, and testability**:

1. **Imports** — Standardized on the `Modules` package (capital M) for consistent behavior on case-sensitive systems (e.g. Linux).
2. **Environment** — `.env.example` added with all required keys: `GEMINI_API_KEY`, `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`, `ADZUNA_COUNTRY`, and optional `LOG_LEVEL`.
3. **Error handling** — Every page and pipeline step uses try/except with `logger.exception` and user-facing messages (`st.error` / `st.warning` / `st.info`). Adzuna errors are decoded into clear guidance.
4. **Logging** — Pipeline stages (parse, search, rank, generate) and all modules use Python logging; level configurable via `LOG_LEVEL`.
5. **Tests** — `tests/` package with pytest:
   - **`test_resume_parser_schema.py`** — Parser output schema and types; mocked LLM for `extract_resume_info` and `extract_keywords`.
   - **`test_job_search_format_jobs.py`** — `format_jobs` output schema and field types using fake Adzuna payloads.
   - **`test_matcher_rank_jobs.py`** — `rank_jobs` length, descending scores, and enriched keys with mocked embeddings.
6. **Dependencies** — `pytest>=7.0.0` added to `requirements.txt`.

All tests run without real API calls and pass from project root with `python -m pytest tests/ -v`.

---

## License

This project is for educational and portfolio use. Respect Adzuna and Google API terms of service and do not commit `.env` or API keys.
