"""UI smoke test: runs render_merit() and render_pathfinder() against a FAKE Streamlit and a FAKE LLM.
Run:  python tests/ui_smoke.py      (no Streamlit, no API key, no internet needed)
It proves the pages are wired correctly end to end. It does not test how they look."""
import contextlib, json, os, sys, tempfile, types, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = []            # everything the page "displays"
OVERRIDES = {}      # widget key -> value returned by the fake widget


class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(FakeSt, name)


class State(dict):
    """Like st.session_state: supports both state['x'] and state.x"""
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v


class FakeSt(types.ModuleType):
    session_state = State()
    secrets = {}
    def __init__(self): super().__init__("streamlit")
    @staticmethod
    def _say(*a, **k): OUT.append(" ".join(str(x) for x in a if isinstance(x, (str, int, float))))
    markdown = write = caption = info = warning = error = success = subheader = title = _say
    @staticmethod
    def divider(*a, **k): pass
    @staticmethod
    def dataframe(data, **k): OUT.append(f"TABLE {json.dumps(data, default=str)[:3000]}")
    @staticmethod
    def metric(label, value, **k): OUT.append(f"METRIC {label}={value}")
    @staticmethod
    def exception(e): raise e
    @staticmethod
    def spinner(*a, **k): return Ctx()
    @staticmethod
    def expander(*a, **k): return Ctx()
    @staticmethod
    def form(*a, **k): return Ctx()
    @staticmethod
    def columns(n, **k): return [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    @staticmethod
    def tabs(names): return [Ctx() for _ in names]
    @staticmethod
    def selectbox(label, options, index=0, key=None, **k): return OVERRIDES.get(key, list(options)[index])
    @staticmethod
    def multiselect(label, options, key=None, **k): return OVERRIDES.get(key, OVERRIDES.get(label, []))
    @staticmethod
    def number_input(label, min_value=None, max_value=None, value=0, step=None, key=None, **k): return OVERRIDES.get(key, OVERRIDES.get(label, value))
    @staticmethod
    def text_input(label, value="", key=None, **k): return OVERRIDES.get(key, value)
    text_area = text_input
    @staticmethod
    def checkbox(label, value=False, key=None, **k): return OVERRIDES.get(key, value)
    @staticmethod
    def button(label, key=None, **k): return OVERRIDES.get(key or label, False)
    form_submit_button = button
    @staticmethod
    def download_button(label, data, *a, **k): OUT.append(f"DOWNLOAD {label} ({len(data)} bytes)")
    @staticmethod
    def file_uploader(*a, **k): return None
    @staticmethod
    def cache_resource(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


fake = FakeSt()
sys.modules["streamlit"] = fake
for name in ("groq", "ddgs", "faiss", "sentence_transformers"):
    m = types.ModuleType(name); m.Groq = object; m.DDGS = object; m.SentenceTransformer = object; sys.modules[name] = m

tmp = tempfile.mkdtemp()
import config; config.DB_PATH = pathlib.Path(tmp) / "t.db"
import db; db.DB_PATH = config.DB_PATH
import memory, rag   # real modules (faiss stubbed)
import agent_system as A
import agent_pages as P
import web_search

FakeSt.session_state["debug_mode"] = True
db.init_db(); db.ensure_student("s1", "Test")
memory.LongTermMemory.add = lambda self, *a, **k: 1          # no embeddings in the sandbox
memory.LongTermMemory.retrieve = lambda self, q, top_k=5: []
rag.load_database_index = lambda: (None, [])
P.load_database_index = lambda: (None, [])

LLM_CALLS = []
def fake_json(prompt, model=None):
    LLM_CALLS.append(prompt[:60])
    if "Extract previous-year CLOSING MERIT" in prompt:
        return {"rows": [
            {"source_index": 0, "year": 2024, "exam": "ECAT", "program": "Electrical Engineering", "institution": "UET Lahore", "closing_merit": 81.25, "list_type": "final"},
            {"source_index": 0, "year": 2024, "exam": "ECAT", "program": "Electrical Engineering", "institution": "UET Lahore", "closing_merit": 99.99, "list_type": "final"},  # invented
        ]}
    if "Path Finder" in prompt and '"answer"' in prompt:
        return {"answer": "Per K1 you must apply via your university.", "evidence_ids": ["K1"], "not_found": False}
    return {"summary": "You are a Pre-Medical student aiming at medicine.",
            "career_paths": [{"title": "Doctor", "why_it_fits": "Interest in medicine.", "steps": ["MBBS"], "evidence_ids": ["K1"]}, {"title": "Fake", "why_it_fits": "", "steps": [], "evidence_ids": ["K99"]}],
            "roadmap": [{"phase": "Next 30 days", "actions": [{"action": "Prepare for MDCAT", "evidence_ids": ["K1"]}, {"action": "Uncited deadline", "evidence_ids": []}]}],
            "web_findings": [{"finding": "A university page mentions admissions.", "evidence_ids": ["W1"]}],
            "missing_information": ["Fees"], "follow_up_questions": ["Which city?"]}
A.generate_json = fake_json
A.generate_text = lambda prompt, **k: "EXPLANATION: your aggregate is shown above; past merits are not guarantees."
web_search_results = [{"title": "UET merit 2024", "url": "https://uet.edu.pk/merit", "snippet": "UET Lahore closed at 81.25 for Electrical Engineering in 2024.", "trusted": "True"}]
A.search_web = lambda q, max_results=5, trusted_only=False: web_search_results

orch = A.Orchestrator.__new__(A.Orchestrator)
orch.memory_agent = types.SimpleNamespace(retrieve=lambda q: [])
orch.merit, orch.pathfinder = A.MeritAgent(), A.PathFinderAgent()

# ---------------------------------------------------------------- 1. Merit page: MDCAT
OVERRIDES.update({"merit_go": True, "m_mdcat_pmdc_matric_o": 1000, "m_mdcat_pmdc_fsc_o": 1000, "m_mdcat_pmdc_test_o": 150,
                  "merit_formula_choice": "MDCAT - PMDC pool (Matric 10 / FSc 40 / MDCAT 50)", "merit_program": "MBBS", "merit_explain": True})
P.render_merit("s1", orch)
text = "\n".join(OUT)
assert "METRIC Your aggregate=" in text, "\n".join(OUT)[-1500:]
assert "Expected cutoff" in text and "King Edward" in text
assert "EXPLANATION" in text
assert "DOWNLOAD" in text
print("merit page OK ->", [l for l in OUT if l.startswith("METRIC")][0])
assert db.merit_results("s1", 1)[0]["exam"] == "MDCAT"

# ---------------------------------------------------------------- 2. Merit page: bad input is handled, ECAT has no invented data
OUT.clear(); FakeSt.session_state.clear()
OVERRIDES.update({"merit_formula_choice": "ECAT - UET Lahore (Matric 17 / FSc 50 / ECAT 33)", "m_ecat_uet_17_50_33_matric_o": 900, "m_ecat_uet_17_50_33_fsc_o": 2000,
                  "m_ecat_uet_17_50_33_test_o": 300, "merit_explain": False})
P.render_merit("s1", orch)
assert any("cannot exceed" in l for l in OUT), OUT[:5]          # FSc 2000 > 1100 is rejected, no crash
OVERRIDES["m_ecat_uet_17_50_33_fsc_o"] = 900
OUT.clear(); P.render_merit("s1", orch)
assert any("No closing-merit data is loaded" in l for l in OUT)
print("ecat page OK (no invented cutoffs, bad marks rejected)")

# ---------------------------------------------------------------- 3. Web extraction keeps only verified rows
accepted, rejected = orch.merit.find_closing_merits("ecat closing merit", "ECAT")
assert len(accepted) == 1 and accepted[0]["closing_merit"] == 81.25 and len(rejected) == 1
print("web extraction gate OK:", accepted[0]["institution"], accepted[0]["closing_merit"], "| rejected:", rejected[0])

# ---------------------------------------------------------------- 4. Path Finder
OUT.clear(); FakeSt.session_state.clear(); OVERRIDES.clear()
OVERRIDES.update({"Find my path": True, "Intermediate stream": "Pre-Medical", "What interests you?": ["Medicine & healthcare"],
                  "FSc / A-Level % (so far)": 58.0, "Family monthly income (PKR)": 40000, "pf_ask": True, "pf_q": "How do I apply for the HEC scholarship?"})
P.render_pathfinder("s1", orch)
text = "\n".join(OUT)
assert "MBBS" in text and "does not meet" not in text.split("Programs")[0]
assert "Doctor" in text and "Fake" not in text                     # uncited/fake career removed
assert "Uncited deadline" not in text                              # uncited roadmap action removed
assert "AI statement(s) were removed" in text
assert "HEC Need Based Scholarship" in text and "Pakistan Medical and Dental Council" in text
assert "Per K1" in text                                           # grounded Q&A
print("pathfinder page OK; dropped notice present; ungrounded items removed")

# ---------------------------------------------------------------- 5. Path Finder survives total LLM failure
def boom(prompt, model=None): raise RuntimeError("Groq down")
A.generate_json = boom
OUT.clear(); FakeSt.session_state.clear(); OVERRIDES["pf_ask"] = False
P.render_pathfinder("s1", orch)
text = "\n".join(OUT)
assert "verified checks below are still complete" in text and "MBBS" in text and "Verification checklist" in text
print("pathfinder degrades gracefully when the AI fails")
print("ALL UI SMOKE TESTS PASSED")
