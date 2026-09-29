"""Data isolation, malformed rollout handling, and metric integrity guards."""
import json
import math
from pathlib import Path
import unittest

from tools.build_synthetic_preferences import build, read_rows
from tools.build_preference_data import input_key
from tools.rl_reward import RewardScorer
from tools.summarize_rl_comparison import calibration

ROOT = Path(__file__).resolve().parents[1]


class PosttrainingTests(unittest.TestCase):
    def test_invalid_rollouts_receive_finite_zero(self):
        scorer = RewardScorer()
        valid = dict(root_cause="过滤器堵塞", confidence=0.8, evidence=[], sop=[])
        for key, bad in (("confidence", float("nan")), ("confidence", float("inf")),
                         ("confidence", True), ("confidence", 1.1), ("confidence", "bad"),
                         ("root_cause", []), ("evidence", 7), ("sop", [None])):
            with self.subTest(key=key, bad=bad):
                score = scorer.score("", json.dumps({**valid, key: bad}), "过滤器堵塞")
                self.assertEqual(score.total, 0)
                self.assertTrue(math.isfinite(score.total))
                self.assertFalse(score.json_ok)

    def test_empty_sop_does_not_match_any_action(self):
        scorer = RewardScorer()
        self.assertEqual(scorer._sop_score("过滤器堵塞", ["", " "]), 0)

    def test_calibration_known_values_and_missing(self):
        rows = [{"conf_raw": 0.8, "correct_raw": True},
                {"conf_raw": 0.8, "correct_raw": False}]
        scores = calibration(rows)
        self.assertAlmostEqual(scores["brier"], 0.34)
        self.assertAlmostEqual(scores["ece_10"], 0.3)
        self.assertIsNone(calibration([])["brier"])

    def test_synthetic_split_and_rescoring(self):
        source = read_rows(ROOT / "data/rl/prompts_gold.jsonl")[:60]
        result, _ = build(source, [], seed=7)
        self.assertEqual(result, build(source, [], seed=7)[0])
        train = {input_key(r["messages"]) for r in result["prefs_train"]}
        held = {input_key(r["messages"]) for r in result["prefs_heldout"]}
        self.assertFalse(train & held)
        self.assertEqual(train, {input_key(r["messages"]) for r in result["prompts_train"]})
        scorer = RewardScorer()
        for row in result["prefs_train"] + result["prefs_heldout"]:
            self.assertEqual(row["source"], "synthetic")
            c = scorer.score(row["prompt"], row["chosen"], row["actual_root_cause"])
            r = scorer.score(row["prompt"], row["rejected"], row["actual_root_cause"])
            self.assertTrue(c.correct)
            self.assertGreaterEqual(c.total - r.total, 0.05)

    def test_leaked_or_duplicated_source_rejected(self):
        source = read_rows(ROOT / "data/rl/prompts_gold.jsonl")[:10]
        with self.assertRaises(ValueError):
            build(source, source[:1])
        with self.assertRaises(ValueError):
            build(source + source[:1], [])


if __name__ == "__main__":
    unittest.main()
