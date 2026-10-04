# Merit Aggregate Agent + Path Finder: tools and platforms

## How the agents avoid inventing information
| Layer | What it does | Where |
|---|---|---|
| Data as files | Formulas, closing merits, programs, scholarships, regulators. Every record has a **source** and a **confidence** | `knowledge/` |
| Deterministic engine | Aggregate maths, bands, eligibility checks. No AI involved | `merit_service.py`, `pathfinder_service.py` |
| Evidence ids | AI may only cite `[K1] [D1] [W1]` evidence | `build_evidence()` |
| Output validation | Any AI statement without a valid citation is dropped (and counted) | `validate_llm_output()` |
| Extraction gate | AI-extracted merit numbers must literally appear in the source text | `verify_extracted_rows()` |
| Honest gaps | Missing data is listed under "Evidence & gaps", never filled in | Path Finder page |

## Open-source tools
| Tool | Used for | License | Status |
|---|---|---|---|
| Streamlit | App UI | Apache-2.0 | already in project |
| FAISS | Search your uploaded prospectuses | MIT | already in project |
| sentence-transformers | Embeddings for document search | Apache-2.0 | already in project |
| ddgs (DuckDuckGo) | Live web search, no API key | MIT | already in project |
| pypdf | Read PDF prospectuses | BSD-3 | already in project |
| ReportLab | PDF reports and roadmaps | BSD | already in project |
| SQLite | Merit history and agent sessions | Public domain | already in project |
| gpt-oss-120b / 20b (open weights) | The language model, served by Groq | Apache-2.0 | already in project |
| pdfplumber | Parse official merit-list PDF tables (UHS/NUMS selection lists) | MIT | recommended next |
| rapidfuzz | Match college names across data sources | MIT | recommended next |

## Platforms and official data sources
| Platform | Purpose |
|---|---|
| Groq API | LLM and Whisper (free tier, key in Streamlit secrets) |
| Streamlit Community Cloud | Hosting. NOTE: its disk is temporary, so move SQLite to Supabase/Postgres or Turso before real users |
| GitHub | Version control for the `knowledge/` files |
| HEC (hec.gov.pk) | University recognition, need-based scholarship |
| PMDC (pmdc.pk) | Medical/dental college recognition, MDCAT rules |
| PEC (pec.org.pk) | Engineering accreditation |
| UHS (uhs.edu.pk), NUMS (numspak.edu.pk) | Official selection lists and closing merits |
| UET / NUST / each university | The only authority on its own formula |

## Keeping the data correct (your most important job)
1. Open `knowledge/merit_formulas.json`: confirm each formula against the current prospectus, fix weights, change `confidence` to `high` and add the official URL.
2. Replace aggregator rows in `knowledge/closing_merit.csv` with official lists (or upload a CSV in the app).
3. Add scholarships, programs and universities to `knowledge/pathfinder_kb.json`.
