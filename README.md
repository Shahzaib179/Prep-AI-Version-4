# PREP AI V4 — Adaptive Multi-Agent Learning Platform

Prep AI V4 is the modular evolution of Prep AI V3.2. It combines RAG, persistent student learning data, long-term memory, adaptive assessment and multiple specialized AI roles.

## This build uses Groq

The LLM provider is now **Groq**. No Gemini SDK or Gemini API key is required.

Default model:

```text
openai/gpt-oss-120b
```

Alternative:

```text
openai/gpt-oss-20b
```

Groq currently lists GPT-OSS 120B and 20B as production models. GPT-OSS 120B is the higher-quality default; 20B is faster and lower-cost.

## Project structure

```text
prep-ai-v4/
├── app.py
├── config.py
├── db.py
├── documents.py
├── rag.py
├── memory.py
├── groq_service.py
├── agent_system.py
├── adaptive.py
├── web_search.py
├── pdf_export.py
├── ui.py
├── requirements.txt
├── README.md
├── .gitignore
├── secrets.example.toml
├── .streamlit/
│   └── config.toml
└── faiss_index/
    ├── database.faiss
    ├── metadata.json
    └── config.json
```

Original database PDFs are not required in GitHub. Only the pre-generated FAISS artifacts are used for Database Learning.

## Python

Use Python 3.12.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Groq API key

Create `.streamlit/secrets.toml` locally:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Never commit this file.

On Streamlit Community Cloud:

1. Open your app.
2. Open **Manage app**.
3. Open **Settings → Secrets**.
4. Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

## LLM model settings

Open **Settings** in Prep AI.

You can choose:

- GPT-OSS 120B — Best quality
- GPT-OSS 20B — Faster / lower cost

Click **Save Settings**.

The selected model is stored in the student's SQLite preferences, restored on the next app run, and placed in Streamlit session state for the current session.

## UI color

The Settings page also stores the selected UI color in the same student preferences record.

## RAG

Personalized Learning supports:

- PDF
- DOCX
- TXT
- MD
- public Google Drive file/folder links

The pipeline is:

```text
Document
  ↓
Extraction
  ↓
Page/source metadata
  ↓
Overlapping chunks
  ↓
Sentence Transformer embeddings
  ↓
FAISS
  ↓
Semantic + keyword hybrid search
  ↓
Groq
```

Database Learning loads:

```text
faiss_index/database.faiss
faiss_index/metadata.json
faiss_index/config.json
```

## Long-term memory

Prep AI stores meaningful learning memories in SQLite and keeps per-student semantic memory artifacts under:

```text
data/memory/<student_id>/
```

## Multi-agent layer

`agent_system.py` contains the specialized roles:

- Memory Agent
- Tutor Agent
- Assessment Agent
- Planner Agent
- Research Agent
- Orchestrator

The deterministic parts of learning remain normal Python: scoring, mastery, revision scheduling, database operations, FAISS retrieval and validation.

## Settings persistence fix

V4 Settings now uses a Streamlit form and stores the LLM model and UI color in SQLite. This avoids the common Streamlit problem where a widget value appears to change but is overwritten on the next rerun. The selected widget values are submitted first, persisted, copied into session state, and then the app reruns.

## SQLite migration

The app automatically adds these columns to an existing V4 `student_preferences` table if they are missing:

```text
llm_model
ui_color
```

It also uses the corrected four-placeholder student insert:

```sql
INSERT OR IGNORE INTO students(id,name,created_at,updated_at)
VALUES (?, ?, ?, ?)
```

## Streamlit Cloud deployment

1. Replace the files in your GitHub repository with this V4 build.
2. Keep your existing `faiss_index` artifacts.
3. Do not upload original database PDFs.
4. Add `GROQ_API_KEY` in Streamlit Cloud Secrets.
5. Deploy/reboot the application.
6. Open **Settings** and select a Groq model.
7. Click **Save Settings**.
8. Test Personalized Learning, Database Learning, Quiz, Tutor, Research Agent and Study Plan.

## Important

Do not commit:

- `.streamlit/secrets.toml`
- API keys
- original database PDFs
- temporary uploaded files
- `__pycache__`
- `.venv`

Run:

```bash
streamlit run app.py
```

## V4 learning/profile fixes

This build includes:

- First-run Student Name + Student ID profile setup.
- All dashboard, quiz, mastery, revision, tutor and research records are scoped to the active Student ID.
- Personalized Learning now includes Topic/Chapter, Mode, Difficulty, Number of MCQs and optional instructions after document processing.
- MCQs are rendered as normal educational cards rather than raw JSON/code.
- AI Tutor is a persistent Streamlit chatbot using `st.chat_message` and `st.chat_input`.
- Both student tutor messages and AI tutor responses are saved to long-term semantic memory and SQLite agent-session history.
- Research requests and research responses are saved to long-term semantic memory and SQLite history.
- Memory page includes Semantic Memory, Tutor Chat, Research Agent and All Agent Sessions tabs.
- Practice My Weak Topics creates a quiz in `st.session_state.quiz` and reruns to the Current Quiz view.
- SQLite connections are explicitly closed after transactions to avoid database-lock errors during quiz/mastery/revision updates.
- Long-term memory clearing removes the old FAISS memory index so stale vectors cannot be reloaded.

### First launch

1. Add `GROQ_API_KEY` to Streamlit Cloud Secrets.
2. Start the app.
3. Enter the student's name and unique Student ID.
4. Use the same Student ID on future sessions to retrieve that student's learning profile and memory.


## How mastery (weak / strong areas) is calculated

Scores live in `mastery_model.py` and are recomputed from the raw answers after every quiz.

- Skipped questions count as a soft miss (half a wrong answer) and are not included in accuracy.
- Recent answers count more than old ones (each older answer is worth 10% less).
- Correct Hard answers earn more credit; wrong Easy answers cost more.
- Small samples are pulled toward 50%, and a topic needs at least 3 answered questions before it is labeled weak or strong.
- Weak < 60% · Developing 60-74% · Strong 75%+ · Mastered 90%+

On first start after upgrading, old stored scores are recalculated automatically (`PRAGMA user_version`).

## Merit Calculator and Path Finder agents

- **Merit Calculator** (`merit_service.py`, `agent_pages.py`): aggregate for MDCAT, NUMS, ECAT, NUST NET, NTS-based or a custom formula; required-test-score calculator; Safe / Target / Reach bands from previous years' closing merits.
- **Path Finder** (`pathfinder_service.py`): programs, eligibility checks, scholarships, accreditation checks and a cited roadmap. Anything the AI says without a valid source is removed.
- Data lives in `knowledge/` (formulas, closing merits, Path Finder knowledge base). See `docs/AGENTS_TOOLS.md`.
- Tests: `python -m unittest tests.test_agents -v` and `python tests/ui_smoke.py` (no API key or internet needed).


## V4.1 additions

### 1 + 2. Memory reset and memory meter
- **Memory** page → *Memory usage* ring chart: memories stored out of the limit, free slots, approximate storage, and a breakdown by type. The Dashboard shows the same ring.
- The limit is `MEMORY_MAX_ITEMS` in `config.py` (default 500 per student). A warning appears at 80 % (`MEMORY_WARN_RATIO`). At 100 % the **least important, then oldest** memories are replaced automatically so the limit is never exceeded.
- *Reset my memory* (bottom of the Memory page) needs an "I understand" tick. Optionally also delete saved Tutor/Research chat history. Quiz results, mastery and revision schedule are never touched.

### 3. Charts instead of percentages
Altair (installed with Streamlit, no new dependency): progress rings (mastery, accuracy, memory), donut charts (correct/incorrect/skipped, memory contents, quiz completion), topic-mastery bars with 60 %/75 % guides, score-trend lines, per-subject bars. Code is in `charts.py`.

### 4. Shared quizzes (many students, same quiz, same time)
The **Published Quizzes** page has four tabs: *Publish* (publishes the current quiz and creates a 6-character code), *Join Quiz* (enter a code), *My Published Quizzes* (codes, attempts, close/re-open) and *Tutor Results* (tutor only). Everyone gets the same questions, each student in their own random order. Each student has **one attempt**, saved under their own Student ID. A student's start time is stored on the server, so reloading the page cannot restart a timed quiz. Students see their own rank and the class average; names are visible to tutors only.

### 5. Number of questions + time limit
Learn (Quiz mode), Weak Topics, Smart Revision, Exam and shared quizzes all take a question count and a time limit in minutes (0 = no limit). A live countdown runs; when time ends the quiz is submitted automatically with the answers selected before the deadline.

### 6. Theme colours
Settings → Appearance: presets (Default, Light, Dark, Sepia, Midnight Blue) or **Custom** background + text colour. Combinations that are unreadable (contrast below 3:1) are refused; below 4.5:1 you get a warning. *Reset to default look* is always available.

### 7. Tutor role and per-student performance
Set `TUTOR_ACCESS_CODE` in Streamlit Secrets. On the first screen choose **Tutor**: only the access code is asked (no name/ID). The tutor then sees only the **Tutor Dashboard** with tabs *Create Quiz*, *Publish*, *My Published Quizzes*, *Quiz Results* and *Student Performance* (class overview, one-student drill-down, CSV downloads). Tutors see performance only, not students' private tutor chats, research or memory contents. Without the secret the Tutor role is disabled.

### Tests
```bash
python -m unittest tests.test_features -v   # database, memory limit/reset, shared quiz, timer maths, theme
python tests/app_pages_smoke.py             # runs every page headlessly against a fake Streamlit
python -m unittest tests.test_agents -v
python tests/ui_smoke.py
```
