# CareerMatch AI – Configuration

Project-wide settings that you can change without editing code are in **`config.json`** in the project root.

## `config.json`

### `matching` (scoring weights)

Controls how the **final job match score** is computed from keyword overlap and semantic (vector) similarity. Weights are normalized to sum to 1.0 if needed.

| Key | Description | Default |
|-----|-------------|---------|
| `keyword_weight` | Weight for keyword-overlap score (resume keywords present in job title/description). Higher = strict ATS-style matching. | `0.6` |
| `vector_weight` | Weight for semantic similarity (embedding cosine similarity). Higher = more “meaning” match, fewer exact keywords required. | `0.4` |

**Example:** For stricter keyword matching use `"keyword_weight": 0.8` and `"vector_weight": 0.2`. For more semantic/role diversity use `0.4` and `0.6`. Restart the app after editing.

### `job_search`

| Key | Description | Default / example |
|-----|-------------|--------------------|
| `fallback_queries` | When the resume-based search returns 0 jobs, the app runs these queries and merges results so users see **multiple role types** (e.g. Data Analyst, Cloud Engineer, DevOps), not only “Software Engineer”. | `["software developer", "data analyst", "cloud engineer", "devops engineer", "data engineer", "backend developer", "full stack developer", "software engineer"]` |
| `fallback_jobs_per_query` | Max jobs to fetch per fallback query before moving to the next. Total jobs is capped by the app (e.g. 100). | `20` |
| `results_per_page` | Adzuna API page size (optional override). | `50` |
| `max_jobs` | Max jobs to request in one run (optional override). | `100` |

**Example:** To add “machine learning engineer” and “product manager” to the fallback list, edit `config.json` and add those strings to `fallback_queries`. Restart the Streamlit app so the new config is loaded.

If `config.json` is missing or invalid, the app uses built-in defaults (see `Modules/job_search.py`).
