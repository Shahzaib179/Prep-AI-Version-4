"""Tests for the v4.1 features (memory limit/reset, shared quizzes, timer maths, roles/tutor view, theme).

Run:  python -m unittest tests.test_features -v
No Streamlit, FAISS, embedding model, API key or internet is needed: light stand-ins are installed first.
"""
import hashlib, os, pathlib, sqlite3, sys, tempfile, types, unittest
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# ---------------------------------------------------------------- stand-ins
class _FlatIP:
    def __init__(self, d): self.d = d; self.v = np.zeros((0, d), dtype="float32")
    @property
    def ntotal(self): return len(self.v)
    def add(self, x): self.v = np.vstack([self.v, np.asarray(x, dtype="float32")])
    def search(self, q, k):
        sims = q @ self.v.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, 1), idx
    def reconstruct_n(self, i, n): return self.v[i:i + n].copy()

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatIP = _FlatIP
def _write(index, path):
    with open(path, "wb") as f: np.save(f, index.v)
def _read(path):
    with open(path, "rb") as f: v = np.load(f)
    idx = _FlatIP(v.shape[1]); idx.add(v); return idx
fake_faiss.write_index, fake_faiss.read_index = _write, _read
sys.modules["faiss"] = fake_faiss

class _St(types.ModuleType):
    session_state = {}
    secrets = {}
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st = _St("streamlit"); sys.modules["streamlit"] = st
for name in ("groq", "ddgs", "sentence_transformers"):
    m = types.ModuleType(name); m.Groq = object; m.DDGS = object; m.SentenceTransformer = object; sys.modules[name] = m

TMP = pathlib.Path(tempfile.mkdtemp())
import config
config.MEMORY_DIR = TMP / "memory"; config.MEMORY_DIR.mkdir()
import db, memory, theme, quiz_runtime  # noqa: E402


def _fake_embed(texts):
    out = []
    for t in texts:
        h = hashlib.md5(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype("float32") - 127.5
        out.append(v / np.linalg.norm(v))
    return np.vstack(out).astype("float32")
memory.embed_texts = _fake_embed

QS = [{"question": f"Q{i}", "options": {"A": "a", "B": "b"}, "answer": "A", "concept": "c", "explanation": "e"} for i in range(4)]
_counter = [0]

def fresh_db():
    _counter[0] += 1
    db.DB_PATH = TMP / f"t{_counter[0]}.db"
    db.init_db()


class Migration(unittest.TestCase):
    def test_old_database_upgrades_without_losing_data(self):
        _counter[0] += 1
        path = TMP / f"old{_counter[0]}.db"
        con = sqlite3.connect(path)
        con.executescript("""
            CREATE TABLE students (id TEXT PRIMARY KEY, name TEXT NOT NULL, exam_name TEXT DEFAULT '', exam_date TEXT, level TEXT DEFAULT 'MDCAT', daily_minutes INTEGER DEFAULT 60, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE student_preferences (student_id TEXT PRIMARY KEY, preferred_difficulty TEXT DEFAULT 'Medium', preferred_language TEXT DEFAULT 'English', explanation_style TEXT DEFAULT 'Detailed', learning_style TEXT DEFAULT 'Examples + Practice', llm_model TEXT DEFAULT 'x', ui_color TEXT DEFAULT 'Blue');
            CREATE TABLE quiz_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, subject TEXT, topic TEXT, total INTEGER, correct INTEGER, incorrect INTEGER, skipped INTEGER, score REAL, difficulty TEXT, created_at TEXT);
            INSERT INTO students VALUES ('old1','Old Student','',NULL,'MDCAT',60,'2026-01-01','2026-01-01');
            INSERT INTO quiz_attempts(student_id,subject,topic,total,correct,incorrect,skipped,score,difficulty,created_at) VALUES ('old1','Biology','Cells',10,7,3,0,70,'Medium','2026-01-02');
        """)
        con.commit(); con.close()
        db.DB_PATH = path
        db.init_db(); db.init_db()          # twice: must be idempotent
        self.assertEqual(db.get_role("old1"), "student")
        h = db.history("old1")
        self.assertEqual(len(h), 1); self.assertEqual(h[0]["score"], 70)
        self.assertEqual(h[0]["timed_out"], 0)


class SharedQuiz(unittest.TestCase):
    def setUp(self): fresh_db()

    def test_many_students_same_quiz_each_recorded_separately(self):
        db.ensure_student("tut", "Tutor"); db.set_role("tut", "tutor")
        for sid, name in (("s1", "Ali"), ("s2", "Sara"), ("s3", "Omar")):
            db.ensure_student(sid, name)
        code = db.create_shared_quiz("tut", "Cells quiz", "Biology", "Cells", "Medium", QS, 600)
        self.assertEqual(len(code), 6)
        quiz = db.get_shared_quiz(code.lower())          # case-insensitive
        self.assertEqual(len(quiz["questions"]), 4); self.assertEqual(quiz["time_limit_sec"], 600)
        db.record_quiz("s1", "Biology", "Cells", QS, {0: "A", 1: "A", 2: "A", 3: "A"}, "Medium", shared_code=code, time_taken_sec=200, time_limit_sec=600)
        db.record_quiz("s2", "Biology", "Cells", QS, {0: "A", 1: "B"}, "Medium", shared_code=code, time_taken_sec=100, time_limit_sec=600)
        res = db.shared_quiz_results(code)
        self.assertEqual([r["name"] for r in res], ["Ali", "Sara"])      # best score first
        self.assertEqual(res[0]["score"], 100); self.assertEqual(res[1]["score"], 25)
        listing = db.list_shared_quizzes()
        self.assertEqual(listing[0]["attempts"], 2)

    def test_one_attempt_per_student(self):
        db.ensure_student("s1", "Ali")
        code = db.create_shared_quiz("s1", "t", "B", "T", "Easy", QS)
        db.record_quiz("s1", "B", "T", QS, {0: "A"}, "Easy", shared_code=code)
        self.assertTrue(db.has_attempted_shared(code, "s1"))
        with self.assertRaises(sqlite3.IntegrityError):
            db.record_quiz("s1", "B", "T", QS, {0: "A"}, "Easy", shared_code=code)
        # normal (non-shared) quizzes can repeat freely
        db.record_quiz("s1", "B", "T", QS, {0: "A"}, "Easy"); db.record_quiz("s1", "B", "T", QS, {0: "A"}, "Easy")

    def test_start_time_is_fixed_server_side(self):
        db.ensure_student("s1", "Ali")
        code = db.create_shared_quiz("s1", "t", "B", "T", "Easy", QS, 60)
        first = db.shared_quiz_start(code, "s1")
        self.assertEqual(db.shared_quiz_start(code, "s1"), first)        # reload cannot restart the clock

    def test_timeout_attempt_is_recorded_but_does_not_touch_mastery(self):
        db.ensure_student("s1", "Ali")
        code = db.create_shared_quiz("s1", "t", "Biology", "Cells", "Easy", QS, 60)
        quiz = db.get_shared_quiz(code)
        db.record_shared_timeout("s1", quiz, 60)
        row = db.shared_quiz_results(code)[0]
        self.assertEqual((row["timed_out"], row["skipped"], row["score"]), (1, 4, 0))
        self.assertEqual(db.topic_mastery("s1"), [])

    def test_closed_flag(self):
        db.ensure_student("s1", "Ali")
        code = db.create_shared_quiz("s1", "t", "B", "T", "Easy", QS)
        db.set_shared_quiz_open(code, False)
        self.assertEqual(db.get_shared_quiz(code)["is_open"], 0)


class TutorView(unittest.TestCase):
    def setUp(self): fresh_db()

    def test_roster_lists_students_with_individual_numbers_and_hides_tutors(self):
        db.ensure_student("tut", "Tutor"); db.set_role("tutor_x" if False else "tut", "tutor")
        db.ensure_student("a", "Alice"); db.ensure_student("b", "Bob"); db.ensure_student("c", "Cara")
        db.record_quiz("a", "Biology", "Cells", QS, {0: "A", 1: "A", 2: "A", 3: "A"}, "Medium")
        db.record_quiz("b", "Biology", "Cells", QS, {0: "B", 1: "B", 2: "B", 3: "B"}, "Medium", timed_out=True, time_limit_sec=60, time_taken_sec=60)
        rows = {r["student_id"]: r for r in db.roster()}
        self.assertEqual(set(rows), {"a", "b", "c"})                      # tutor not listed
        self.assertEqual(rows["a"]["avg_score"], 100); self.assertEqual(rows["a"]["accuracy"], 100)
        self.assertEqual(rows["b"]["avg_score"], 0); self.assertEqual(rows["b"]["timeouts"], 1)
        self.assertEqual(rows["c"]["quizzes"], 0)
        trend = db.student_quiz_trend("a"); self.assertEqual(len(trend), 1)
        self.assertEqual(db.subject_breakdown("a")[0]["accuracy"], 100)

    def test_role_validation(self):
        db.ensure_student("x", "X")
        with self.assertRaises(ValueError): db.set_role("x", "admin")

    def test_students_cannot_see_each_others_data(self):
        db.ensure_student("a", "Alice"); db.ensure_student("b", "Bob")
        db.record_quiz("a", "Biology", "Cells", QS, {0: "A"}, "Medium")
        self.assertEqual(db.history("b"), []); self.assertEqual(len(db.history("a")), 1)


class MemoryLimits(unittest.TestCase):
    def setUp(self):
        fresh_db(); db.ensure_student("m1", "Mem")
        self._old = memory.MEMORY_MAX_ITEMS; memory.MEMORY_MAX_ITEMS = 5
        import shutil; shutil.rmtree(config.MEMORY_DIR / "m1", ignore_errors=True)
    def tearDown(self): memory.MEMORY_MAX_ITEMS = self._old

    def _fill(self, m, n, importance=0.5, tag="mem"):
        for i in range(n): m.add(f"{tag} number {i} about photosynthesis", "learning", importance=importance)

    def test_usage_reports_count_limit_percent_and_bytes(self):
        m = memory.LongTermMemory("m1"); self._fill(m, 2)
        u = m.usage()
        self.assertEqual((u["used"], u["limit"], u["free"], u["percent"]), (2, 5, 3, 40.0))
        self.assertFalse(u["near_limit"]); self.assertGreater(u["text_bytes"], 0); self.assertGreater(u["disk_bytes"], 0)
        self._fill(m, 2, tag="more")
        self.assertTrue(m.usage()["near_limit"])                          # 4/5 = 80%

    def test_cap_is_enforced_and_least_important_goes_first(self):
        m = memory.LongTermMemory("m1")
        m.add("IMPORTANT fact the student must keep", "learning", importance=0.99)
        self._fill(m, 4, importance=0.2, tag="low")
        self.assertEqual(m.count(), 5)
        m.add("A brand new memory about genetics", "learning", importance=0.6)
        self.assertEqual(m.count(), 5)                                    # never above the limit
        contents = [x["content"] for x in m.recent(10)]
        self.assertTrue(any("IMPORTANT" in c for c in contents))          # high importance survived
        self.assertTrue(any("brand new" in c for c in contents))
        self.assertFalse(any("low number 0" in c for c in contents))      # oldest low-importance evicted
        self.assertEqual(m.index.ntotal, len(m.metadata))                 # vector index stays in sync
        again = memory.LongTermMemory("m1")                               # and survives a reload from disk
        self.assertEqual((again.index.ntotal, len(again.metadata)), (5, 5))
        self.assertTrue(again.retrieve("genetics", 3))

    def test_reset_clears_memory_but_keeps_quiz_results(self):
        m = memory.LongTermMemory("m1"); self._fill(m, 3)
        db.save_agent_session("m1", "Tutor Agent", "hi", "hello")
        db.record_quiz("m1", "Biology", "Cells", QS, {0: "A"}, "Medium")
        out = memory.reset_memory("m1", include_conversations=False)
        self.assertEqual(out, {"memories_removed": 3, "conversations_removed": 0})
        fresh = memory.LongTermMemory("m1")
        self.assertEqual((fresh.count(), fresh.index, fresh.usage()["used"]), (0, None, 0))
        self.assertEqual(len(db.get_agent_sessions("m1")), 1)             # chats kept
        self.assertEqual(len(db.history("m1")), 1)                        # quizzes untouched
        out = memory.reset_memory("m1", include_conversations=True)
        self.assertEqual(out["conversations_removed"], 1); self.assertEqual(db.get_agent_sessions("m1"), [])
        self.assertEqual(len(db.history("m1")), 1)

    def test_reset_only_affects_the_one_student(self):
        db.ensure_student("m2", "Other")
        a = memory.LongTermMemory("m1"); b = memory.LongTermMemory("m2")
        self._fill(a, 2); self._fill(b, 3, tag="other")
        memory.reset_memory("m1", True)
        self.assertEqual(memory.LongTermMemory("m2").count(), 3)


class QuizTimer(unittest.TestCase):
    def test_minutes_conversion(self):
        f = quiz_runtime.minutes_to_seconds
        self.assertEqual((f(0), f(None), f(-5), f("x"), f(30), f(1.5)), (0, 0, 0, 0, 1800, 90))
        self.assertEqual(f(10_000), quiz_runtime.QUIZ_MAX_MINUTES * 60)   # capped

    def test_remaining_and_expiry(self):
        r, e = quiz_runtime.remaining_seconds, quiz_runtime.is_expired
        self.assertIsNone(r(1000, 0, now=1500)); self.assertFalse(e(1000, 0, now=10**9))   # untimed never expires
        self.assertEqual(r(1000, 600, now=1100), 500)
        self.assertEqual(r(1000, 600, now=5000), 0)
        self.assertFalse(e(1000, 600, now=1599)); self.assertTrue(e(1000, 600, now=1600))
        self.assertEqual(quiz_runtime.elapsed_seconds(1000, 600, now=9999), 600)           # clipped to the limit

    def test_clock_format(self):
        c = quiz_runtime.clock
        self.assertEqual((c(None), c(0), c(65), c(3725)), ("No limit", "00:00", "01:05", "1:02:05"))

    def test_iso_roundtrip_matches_utc_now(self):
        stamp = datetime.utcnow().isoformat()
        self.assertLess(abs(quiz_runtime.iso_to_epoch(stamp) - datetime.utcnow().replace(tzinfo=None).timestamp() - (datetime.now().astimezone().utcoffset().total_seconds())), 5)
        old = (datetime.utcnow() - timedelta(seconds=700)).isoformat()
        self.assertTrue(quiz_runtime.is_expired(quiz_runtime.iso_to_epoch(old), 600))      # started 700s ago, 600s limit


class Theme(unittest.TestCase):
    def test_contrast(self):
        self.assertEqual(theme.contrast_ratio("#000000", "#FFFFFF"), 21.0)
        self.assertEqual(theme.contrast_ratio("#FFFFFF", "#FFFFFF"), 1.0)

    def test_validation_blocks_unreadable_pairs(self):
        self.assertEqual(theme.validate_theme("#FFFFFF", "#FFFFFE")[0], "block")
        self.assertEqual(theme.validate_theme("#FFFFFF", "#888888")[0], "warn")      # ~3.5:1
        self.assertEqual(theme.validate_theme("#0E1117", "#F3F4F6")[0], "ok")
        self.assertEqual(theme.validate_theme("red", "#000000")[0], "block")
        for name, pair in config.THEME_PRESETS.items():
            if pair: self.assertEqual(theme.validate_theme(*pair)[0], "ok", name)    # every preset is readable

    def test_resolve(self):
        self.assertIsNone(theme.resolve_theme("Default", "#111111", "#EEEEEE"))
        self.assertEqual(theme.resolve_theme("Dark", "", ""), ("#0E1117", "#F3F4F6"))
        self.assertEqual(theme.resolve_theme("Custom", "#102030", "#f0f0f0"), ("#102030", "#F0F0F0"))
        self.assertIsNone(theme.resolve_theme("Custom", "bad", "#000000"))

    def test_css(self):
        plain = theme.build_css("#2563EB"); self.assertNotIn("user theme", plain)
        css = theme.build_css("#2563EB", "#102030", "#F0F0F0")
        self.assertIn("#102030", css); self.assertIn("#F0F0F0", css); self.assertIn("prep-hero", css)

    def test_preferences_persist_per_student(self):
        fresh_db(); db.ensure_student("t1", "A"); db.ensure_student("t2", "B")
        db.update_preferences("t1", theme_preset="Custom", theme_bg="#102030", theme_text="#F0F0F0")
        self.assertEqual(db.get_preferences("t1")["theme_bg"], "#102030")
        self.assertEqual(db.get_preferences("t2")["theme_preset"], "Default")


if __name__ == "__main__":
    unittest.main()
