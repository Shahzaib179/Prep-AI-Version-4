"""Headless run of app.py: every page is executed against a FAKE Streamlit (no browser, no API key).
Proves the new code is wired correctly (no NameError / bad arguments). It cannot judge how pages LOOK.
Run:  python tests/app_pages_smoke.py
"""
import contextlib, os, pathlib, sys, tempfile, types
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT, OVR = [], {}

class Rerun(Exception): pass
class Stop(Exception): pass
class State(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(FakeSt, n)

class FakeSt(types.ModuleType):
    session_state = State(); secrets = {}
    def __init__(self): super().__init__("streamlit")
    @staticmethod
    def _say(*a, **k): OUT.append(" ".join(str(x) for x in a if isinstance(x, (str, int, float))))
    markdown = write = caption = info = warning = error = success = subheader = title = code = _say
    set_page_config = divider = staticmethod(lambda *a, **k: None)
    @staticmethod
    def stop(): raise Stop()
    @staticmethod
    def rerun(*a, **k): raise Rerun()
    @staticmethod
    def exception(e): raise e
    @staticmethod
    def fragment(*a, **k): return (lambda f: f) if not (a and callable(a[0])) else a[0]
    @staticmethod
    def cache_resource(*a, **k): return (lambda f: f) if not (a and callable(a[0])) else a[0]
    sidebar = Ctx()
    spinner = expander = form = container = chat_message = staticmethod(lambda *a, **k: Ctx())
    @staticmethod
    def columns(n, **k): return [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    @staticmethod
    def tabs(names): return [Ctx() for _ in names]
    @staticmethod
    def metric(label, value, **k): OUT.append(f"METRIC {label}={value}")
    @staticmethod
    def progress(v, text=None, **k): OUT.append(f"PROGRESS {text}")
    @staticmethod
    def altair_chart(c, **k): OUT.append("CHART")
    @staticmethod
    def dataframe(d, **k): OUT.append("TABLE")
    @staticmethod
    def download_button(label, data, *a, **k): OUT.append(f"DOWNLOAD {label}")
    @staticmethod
    def radio(label, options=(), index=0, key=None, **k):
        opts = list(options); return OVR.get(key or label, opts[index] if (opts and index is not None) else None)
    @staticmethod
    def selectbox(label, options, index=0, key=None, **k): return OVR.get(key or label, list(options)[index or 0])
    @staticmethod
    def multiselect(label, options, key=None, default=None, **k): return OVR.get(key or label, default or [])
    @staticmethod
    def slider(label, min_value=0, max_value=100, value=None, key=None, **k): return OVR.get(key or label, value if value is not None else min_value)
    @staticmethod
    def number_input(label, min_value=None, max_value=None, value=0, *a, key=None, **k): return OVR.get(key or label, value)
    @staticmethod
    def date_input(label, value=None, key=None, **k): return OVR.get(key or label, value)
    @staticmethod
    def text_input(label, value="", key=None, **k): return OVR.get(key or label, value)
    text_area = text_input
    @staticmethod
    def color_picker(label, value="#000000", key=None, **k): return OVR.get(key or label, value)
    @staticmethod
    def checkbox(label, value=False, key=None, **k): return OVR.get(key or label, FakeSt.session_state.get(key, value) if key else value)
    @staticmethod
    def button(label, key=None, **k): return OVR.get(key or label, False)
    form_submit_button = button
    @staticmethod
    def file_uploader(*a, **k): return None
    @staticmethod
    def audio_input(*a, **k): return None
    @staticmethod
    def chat_input(*a, **k): return None

fake = FakeSt(); sys.modules["streamlit"] = fake
for n in ("groq", "ddgs", "faiss", "sentence_transformers", "gdown", "gtts", "altair"):
    m = mock.MagicMock(name=n); sys.modules[n] = m
import config
tmp = pathlib.Path(tempfile.mkdtemp()); config.DB_PATH = tmp / "t.db"; config.MEMORY_DIR = tmp / "mem"; config.MEMORY_DIR.mkdir()
import memory as _m
_m.embed_texts = lambda texts: __import__("numpy").ones((len(texts), 4), dtype="float32")

SRC = (ROOT / "app.py").read_text()
QS = [{"question": f"Q{i}", "options": {"A": "a", "B": "b"}, "answer": "A", "concept": "c", "explanation": "e", "difficulty": "Medium"} for i in range(3)]

def run(page, **state):
    """Execute app.py once as a signed-in user viewing `page`."""
    OUT.clear()
    ss = fake.session_state
    ss.update({"profile_complete": True, "student_id": "stu1", "student_name": "Ali", **state})
    OVR["Navigation"] = page
    g = {"__name__": "__main__", "__file__": str(ROOT / "app.py")}
    try: exec(compile(SRC, "app.py", "exec"), g)
    except (Rerun, Stop): pass
    return g

failures = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <- {detail}"))
    if not cond: failures.append(name)

import db
# ---- sign in as a new student, then visit every page
g = run("Dashboard")
check("dashboard renders (rings + donut charts)", OUT.count("CHART") >= 3, OUT[:5])
for p in ("Learn", "Practice", "Exam", "Published Quizzes", "AI Tutor", "Merit Calculator", "Path Finder", "Study Plan", "Memory", "History", "Settings"):
    try:
        run(p); check(f"page '{p}' runs", True)
    except Exception as e:
        check(f"page '{p}' runs", False, repr(e))
run("Memory"); check("memory page shows usage meter", any("Memories stored" in o for o in OUT))
run("Settings"); check("settings has Appearance", any("Appearance" in o for o in OUT))

# ---- tutor navigation + dashboard
run("Dashboard", is_tutor=False); 
check("non-tutor cannot open tutor page", not any("Tutor Dashboard" in o and "METRIC" in o for o in OUT))
db.ensure_student("stu2", "Sara")
db.record_quiz("stu2", "Biology", "Cells", QS, {0: "A", 1: "A", 2: "B"}, "Medium")
run("Tutor Dashboard", is_tutor=True)
check("tutor dashboard shows class table + charts", "TABLE" in OUT and "CHART" in OUT and any(o.startswith("METRIC Students") for o in OUT), OUT[:8])
run("Tutor Dashboard", is_tutor=False); check("tutor page refuses a non-tutor", any("Tutor access is required" in o for o in OUT), OUT[:5])

# ---- timed practice quiz: gate -> start -> auto-submit at expiry
import time
ss = fake.session_state
g = run("Practice", quiz=None)
quiz_mod = run("Practice")  # builds helpers in globals
new_quiz = quiz_mod["new_quiz"]
ss["quiz"] = new_quiz([dict(q) for q in QS], "Biology", "Cells", "Medium", [], 120)
run("Practice"); check("timed quiz waits for Start click", ss["quiz"]["started_at"] is None and any("Start Quiz" in o or "limit" in o for o in OUT), OUT[:6])
OVR["start_" + ss["quiz"]["uid"]] = True; run("Practice"); OVR.clear()
check("Start click begins the clock", ss["quiz"]["started_at"] is not None)
ss["quiz"]["answers"] = {0: "A", 1: "B"}; ss["quiz"]["answer_ts"] = {0: ss["quiz"]["started_at"] + 1, 1: ss["quiz"]["started_at"] + 2}
run("Practice"); check("quiz still open before the deadline", ss["quiz"]["result"] is None)
ss["quiz"]["started_at"] -= 500                                   # pretend 500s passed (limit 120s)
ss["quiz"]["answer_ts"] = {0: ss["quiz"]["started_at"] + 5, 1: ss["quiz"]["started_at"] + 400}   # Q2 picked AFTER the deadline
run("Practice")
res = ss["quiz"]["result"]
check("expired quiz is auto-submitted", res is not None and res["timed_out"] is True, res)
check("late answer is ignored, early one kept", res and res["correct"] == 1 and res["skipped"] == 2, res)
check("result saved in database as timed out", db.history("stu1")[0]["timed_out"] == 1 if db.history("stu1") else False)
run("Practice"); check("result screen shows score charts", OUT.count("CHART") >= 2, OUT[:6])
n_before = len(db.history("stu1")); run("Practice"); check("re-render never double-saves", len(db.history("stu1")) == n_before)

# ---- untimed quiz needs no Start click and manual submit works
ss["quiz"] = new_quiz([dict(q) for q in QS], "Biology", "Cells", "Medium", [], 0)
uid = ss["quiz"]["uid"]; run("Practice")
check("untimed quiz starts immediately", ss["quiz"]["started_at"] is not None)
ss["quiz"]["answers"] = {0: "A", 1: "A", 2: "A"}; OVR["submit_" + uid] = True; run("Practice"); OVR.clear()
check("manual submit scores 3/3", ss["quiz"]["result"] and ss["quiz"]["result"]["correct"] == 3 and not ss["quiz"]["result"]["timed_out"])

# ---- share + join
ss["quiz"] = new_quiz([dict(q) for q in QS], "Biology", "Cells", "Medium", [], 0)
uid = ss["quiz"]["uid"]; OVR["share_btn_" + uid] = True; run("Practice"); OVR.clear()
code = ss["quiz"].get("share_code_created")
check("share code created", bool(code) and len(code) == 6, code)
ss["student_id"] = "stu3"; ss["student_name"] = "Omar"; ss["shared_quiz"] = None
OVR["Quiz code"] = code; OVR["Open quiz"] = True; run("Published Quizzes", student_id="stu3", student_name="Omar"); OVR.clear()
check("student joins by code", ss["shared_quiz"] is not None and ss["shared_quiz"]["shared_code"] == code)
uid = ss["shared_quiz"]["uid"]; OVR["start_" + uid] = True; run("Published Quizzes", student_id="stu3", student_name="Omar"); OVR.clear()
ss["shared_quiz"]["answers"] = {0: "A", 1: "A", 2: "B"}; OVR["submit_" + uid] = True; run("Published Quizzes", student_id="stu3", student_name="Omar"); OVR.clear()
check("shared attempt saved under joiner's ID", [r["student_id"] for r in db.shared_quiz_results(code)] == ["stu3"])
ss["shared_quiz"] = None; OVR["Quiz code"] = code; OVR["Open quiz"] = True; run("Published Quizzes", student_id="stu3", student_name="Omar"); OVR.clear()
check("second attempt refused", ss["shared_quiz"] is None and any("already completed" in o for o in OUT), OUT[:4])

# ---- tutor login shows only the access code; tutor sees only the Tutor Dashboard
import hashlib
config.get_tutor_code = lambda: "secret-code"
fields = []
orig_ti, orig_form = FakeSt.text_input, FakeSt.form_submit_button
FakeSt.text_input = staticmethod(lambda label, value="", key=None, **k: (fields.append(label), OVR.get(key or label, value))[1])
OVR["login_role"] = "Tutor"; OVR["Tutor access code"] = "wrong"; OVR["Open Tutor Dashboard"] = True
OUT.clear(); ss.update({"profile_complete": False, "student_id": ""}); g = {"__name__": "__main__", "__file__": str(ROOT / "app.py")}
try: exec(compile(SRC, "app.py", "exec"), g)
except (Rerun, Stop): pass
check("tutor login asks ONLY for the access code", fields == ["Tutor access code"], fields)
check("wrong tutor code refused", any("Incorrect tutor access code" in o for o in OUT) and not ss["profile_complete"], OUT[:4])
OVR["Tutor access code"] = "secret-code"; fields.clear(); ss.update({"profile_complete": False, "student_id": ""})
try: exec(compile(SRC, "app.py", "exec"), g)
except (Rerun, Stop): pass
check("right tutor code signs in as tutor", ss["profile_complete"] and ss["is_tutor"] and ss["student_id"].startswith("tutor_"), dict(ss))
TUTOR_ID = ss["student_id"]
OVR.clear(); FakeSt.text_input = staticmethod(orig_ti)
navs = []
orig_radio = FakeSt.radio
FakeSt.radio = staticmethod(lambda label, options=(), index=0, key=None, **k: (navs.append(list(options)) if label == "Navigation" else None, OVR.get(key or label, list(options)[index] if options else None))[1])
run("Tutor Dashboard", is_tutor=True, student_id=TUTOR_ID, student_name="Tutor")
check("tutor navigation is only the Tutor Dashboard", navs and navs[0] == ["Tutor Dashboard"], navs)
navs.clear(); run("Dashboard", is_tutor=False, student_id="stu1", student_name="Ali")
check("student navigation has Published Quizzes, no Join Quiz/Tutor Dashboard", "Published Quizzes" in navs[0] and "Join Quiz" not in navs[0] and "Tutor Dashboard" not in navs[0], navs)
FakeSt.radio = staticmethod(orig_radio)

# ---- tutor publishes from the dashboard; students get their own question order
tid = TUTOR_ID
ss["quiz"] = new_quiz([dict(q, question=f"Q{i}") for i, q in enumerate(QS * 4)], "Biology", "Cells", "Medium", [], 0)
uid = ss["quiz"]["uid"]; OVR["share_btn_" + uid] = True; run("Tutor Dashboard", is_tutor=True, student_id=tid, student_name="Tutor"); OVR.clear()
tcode = ss["quiz"].get("share_code_created")
check("tutor publishes a quiz from the dashboard", bool(tcode) and [q["code"] for q in db.list_shared_quizzes(tid)] == [tcode], (tcode, tid, [q["code"] for q in db.list_shared_quizzes()]))
orders = []
for sid in ("a1", "b2", "c3", "d4"):
    ss["shared_quiz"] = None; OVR["Quiz code"] = tcode; OVR["Open quiz"] = True
    run("Published Quizzes", student_id=sid, student_name=sid, is_tutor=False); OVR.clear()
    orders.append(tuple(q["question"] for q in ss["shared_quiz"]["questions"]))
check("same questions for everyone", len({tuple(sorted(o)) for o in orders}) == 1)
check("each student gets a different order", len(set(orders)) > 1, orders)
ss["shared_quiz"] = None
run("Published Quizzes", student_id="a1", student_name="a1", is_tutor=False)
check("students see results as locked", any("visible only to the tutor" in o for o in OUT), OUT[:6])

# ---- memory reset button
from memory import LongTermMemory
ss["student_id"] = "stu1"; ss["student_name"] = "Ali"
for i in range(3): LongTermMemory("stu1").add(f"memory item {i} about cells", "learning")
check("memory exists before reset", LongTermMemory("stu1").count() == 3)
# the button callback is passed to st.button(on_click=...): capture and invoke it
captured = {}
orig_button = FakeSt.button
FakeSt.button = staticmethod(lambda label, key=None, **k: captured.setdefault(label, k.get("on_click")) and False)
ss["reset_confirm"] = True; ss["reset_incl_chats"] = False
run("Memory"); FakeSt.button = orig_button
cb = captured.get("Reset my memory now"); check("reset button wired with callback", callable(cb))
cb(); check("callback empties memory", LongTermMemory("stu1").count() == 0)
check("confirmation box un-ticks itself", ss["reset_confirm"] is False)
run("Memory"); check("page tells the user what was removed", any("Memory reset" in o for o in OUT), OUT[:5])

# ---- theme applied
db.update_preferences("stu1", theme_preset="Dark")
for k in ("theme_preset", "theme_bg", "theme_text"): ss.pop(k, None)
run("Dashboard"); check("saved Dark theme loads and reaches charts", ss["theme_pair"] == ("#0E1117", "#F3F4F6"), ss.get("theme_pair"))

print("\nFAILED:" if failures else "\nALL PASSED", failures or "")
sys.exit(1 if failures else 0)
