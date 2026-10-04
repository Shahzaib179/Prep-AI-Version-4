"""Run with:  python -m unittest tests.test_agents -v   (no Streamlit / API key needed)"""
import unittest

import merit_service as ms
import pathfinder_service as pf


class MeritTests(unittest.TestCase):
    def test_all_formula_weights_sum_to_100(self):
        for f in ms.load_formulas():
            self.assertAlmostEqual(sum(c["weight"] for c in f["components"]), 100)

    def test_mdcat_example(self):
        f = ms.get_formula("mdcat_pmdc")
        r = ms.calculate_aggregate(f, {"matric": (1050, 1100), "fsc": (1020, 1100), "test": (153, 180)})
        # 95.4545*.10 + 92.7273*.40 + 85*.50 = 9.5455 + 37.0909 + 42.5
        self.assertAlmostEqual(r["aggregate"], 89.1364, places=3)

    def test_legacy_uet_example_matches_source(self):
        f = ms.get_formula("ecat_uet_25_45_30")
        r = ms.calculate_aggregate(f, {"matric": (950, 1100), "fsc": (1000, 1100), "test": (280, 400)})
        self.assertAlmostEqual(r["aggregate"], 83.50, places=1)  # the source's worked example

    def test_validation(self):
        f = ms.get_formula("mdcat_pmdc")
        with self.assertRaises(ValueError):
            ms.calculate_aggregate(f, {"matric": (1200, 1100), "fsc": (900, 1100), "test": (100, 180)})
        with self.assertRaises(ValueError):
            ms.calculate_aggregate(f, {"matric": (900, 0), "fsc": (900, 1100), "test": (100, 180)})
        with self.assertRaises(ValueError):
            ms.custom_formula(10, 40, 40)  # 90 != 100

    def test_required_score_roundtrip(self):
        f = ms.get_formula("mdcat_pmdc")
        marks = {"matric": (1000, 1100), "fsc": (1000, 1100)}
        need = ms.required_test_score(f, marks, target=93.0, test_total=180)
        r = ms.calculate_aggregate(f, {**marks, "test": (need["needed_marks"], 180)})
        self.assertAlmostEqual(r["aggregate"], 93.0, delta=0.2)
        impossible = ms.required_test_score(f, {"matric": (600, 1100), "fsc": (600, 1100)}, 99, 180)
        self.assertFalse(impossible["feasible"])

    def test_eligibility_uses_only_written_rules(self):
        f = ms.get_formula("mdcat_pmdc")
        checks = ms.eligibility_checks(f, {"test": (90, 180)}, "MBBS")  # 50% < 55%
        self.assertEqual([c["status"] for c in checks], ["fail"])
        self.assertEqual(ms.eligibility_checks(ms.get_formula("ecat_uet_17_50_33"), {}, None), [])

    def test_recommend_bands_and_data_hygiene(self):
        rows = ms.load_closing_merits(include_user=False)
        recs = ms.recommend(94.0, rows, exam="MDCAT", program="MBBS")
        self.assertTrue(recs)
        kemu = next(r for r in recs if "King Edward" in r["institution"])
        self.assertEqual(kemu["list_type"], "final")          # final series preferred over first-list
        self.assertEqual(kemu["years_used"], [2024, 2025])     # first-list years not mixed in
        self.assertEqual(kemu["confidence"], "low")            # aggregator-only data -> low
        self.assertIn("Unverified", kemu["notes"])
        fjmu = next(r for r in recs if "Fatima Jinnah" in r["institution"])
        self.assertEqual(fjmu["list_type"], "first")
        self.assertIn("FIRST-list", fjmu["notes"])
        # 80% should never be 'Safe' anywhere
        low = ms.recommend(80.0, rows, exam="MDCAT", program="MBBS")
        self.assertTrue(all(r["band"] in {"Reach", "Unlikely"} for r in low))
        # sorted: Safe first
        order = [ms.BAND_ORDER[r["band"]] for r in recs]
        self.assertEqual(order, sorted(order))

    def test_no_data_means_no_recommendation(self):
        rows = ms.load_closing_merits(include_user=False)
        self.assertEqual(ms.recommend(85.0, rows, exam="ECAT"), [])  # nothing invented for ECAT

    def test_csv_import_and_extraction_gate(self):
        good = ms.template_csv()
        rows, errors = ms.parse_closing_csv(good)
        self.assertEqual((len(rows), errors), (1, []))
        _, errors = ms.parse_closing_csv("year,exam,institution\n2025,ECAT,X\n")
        self.assertTrue(errors)
        text = "Last merit for Allama Iqbal Medical College is 92.9364 per cent in 2023."
        ok, bad = ms.verify_extracted_rows([
            {"year": 2023, "exam": "MDCAT", "program": "MBBS", "institution": "Allama Iqbal Medical College", "closing_merit": 92.9364},
            {"year": 2023, "exam": "MDCAT", "program": "MBBS", "institution": "Allama Iqbal Medical College", "closing_merit": 95.1111},  # invented number
            {"year": 2023, "exam": "MDCAT", "program": "MBBS", "institution": "Imaginary Medical College", "closing_merit": 92.9364},  # invented name
        ], text)
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(bad), 2)


class PathFinderTests(unittest.TestCase):
    def setUp(self):
        self.kb = pf.load_kb()
        self.profile = {"stream": "Pre-Medical", "interests": ["Medicine & healthcare"], "fsc_pct": 58.0, "test_pct": 70.0, "family_income": 40000, "province": "Punjab"}

    def test_matching_and_eligibility(self):
        programs = pf.match_programs(self.kb, self.profile)
        ids = {p["id"] for p in programs}
        self.assertTrue({"prog_mbbs", "prog_bds", "prog_pharmd"} <= ids)
        self.assertNotIn("prog_engineering", ids)
        mbbs = next(p for p in programs if p["id"] == "prog_mbbs")
        checks = pf.check_program_eligibility(mbbs, self.profile)
        by_rule = {c["rule"]: c["status"] for c in checks}
        self.assertEqual(by_rule["Stream: Pre-Medical"], "meets")
        self.assertTrue(any(c["status"] == "does not meet" and "60%" in c["rule"] for c in checks))  # 58 < 60 (NUMS)
        self.assertTrue(any(c["status"] == "meets" and "55%" in c["rule"] for c in checks))          # 70 >= 55

    def test_wrong_stream_and_unknown_inputs(self):
        eng = next(p for p in self.kb["programs"] if p["id"] == "prog_engineering")
        checks = pf.check_program_eligibility(eng, {"stream": "Pre-Medical"})
        self.assertEqual(checks[0]["status"], "does not meet")
        cs = next(p for p in self.kb["programs"] if p["id"] == "prog_computing")
        self.assertEqual(pf.check_program_eligibility(cs, {"stream": "Pre-Medical"})[0]["status"], "cannot check")
        mbbs = next(p for p in self.kb["programs"] if p["id"] == "prog_mbbs")
        self.assertTrue(any(c["status"] == "cannot check" for c in pf.check_program_eligibility(mbbs, {"stream": "Pre-Medical"})))

    def test_scholarship_income_rule(self):
        s = pf.check_scholarships(self.kb, {"family_income": 40000})[0]
        self.assertEqual(s["checks"][0]["status"], "meets")
        s = pf.check_scholarships(self.kb, {"family_income": 90000})[0]
        self.assertEqual(s["checks"][0]["status"], "does not meet")
        s = pf.check_scholarships(self.kb, {})[0]
        self.assertEqual(s["checks"][0]["status"], "cannot check")

    def test_validation_drops_uncited_and_fake_ids(self):
        programs = pf.match_programs(self.kb, self.profile)
        ev = pf.build_evidence(self.kb, programs, pf.check_scholarships(self.kb, {}), pf.regulator_checks(self.kb, programs), pf.careers_for(self.kb, programs),
                               [{"filename": "prospectus.pdf", "page": 3, "text": "Fee 100000"}], [{"title": "t", "url": "https://x.edu.pk", "snippet": "s", "trusted": "True"}])
        ids = {e["id"] for e in ev}
        self.assertTrue({"K1", "D1", "W1"} <= ids)
        data = {
            "summary": "ok",
            "career_paths": [
                {"title": "Doctor", "why_it_fits": "x", "steps": ["a"], "evidence_ids": ["K1"]},
                {"title": "Made up", "why_it_fits": "x", "steps": [], "evidence_ids": ["K99"]},     # fake id
                {"title": "No cite", "why_it_fits": "x", "steps": [], "evidence_ids": []},           # no citation
            ],
            "roadmap": [{"phase": "Next 30 days", "actions": [{"action": "Do A", "evidence_ids": ["D1"]}, {"action": "Invented deadline 5 Nov", "evidence_ids": []}]}],
            "web_findings": [{"finding": "f", "evidence_ids": ["W1"]}, {"finding": "g", "evidence_ids": ["W7"]}],
            "missing_information": ["Fees"], "follow_up_questions": ["a", "b", "c", "d"],
        }
        clean, dropped = pf.validate_llm_output(data, ev)
        self.assertEqual([c["title"] for c in clean["career_paths"]], ["Doctor"])
        self.assertEqual(len(clean["roadmap"][0]["actions"]), 1)
        self.assertEqual(len(clean["web_findings"]), 1)
        self.assertEqual(dropped, 4)
        self.assertEqual(len(clean["follow_up_questions"]), 3)
        self.assertEqual(pf.validate_llm_output("garbage", ev)[0]["career_paths"], [])

    def test_qa_never_passes_ungrounded_answer(self):
        ev = [{"id": "K1", "kind": "knowledge", "title": "t", "text": "x", "url": "", "source_type": "official", "confidence": "high"}]
        self.assertTrue(pf.validate_qa({"answer": "Fee is 5 lakh", "evidence_ids": [], "not_found": False}, ev)["not_found"])
        self.assertTrue(pf.validate_qa({"answer": "Fee is 5 lakh", "evidence_ids": ["K9"], "not_found": False}, ev)["not_found"])
        self.assertFalse(pf.validate_qa({"answer": "From K1", "evidence_ids": ["K1"], "not_found": False}, ev)["not_found"])


if __name__ == "__main__":
    unittest.main()
