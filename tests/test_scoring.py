"""Synthetic protocol fixtures only; these are not benchmark observations."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from react_ablation.scoring import score_checkpoint, score_directory


def error_checkpoint():
    return {
        "checkpoint_id": "test-error", "game_id": "synthetic-game", "kind": "error",
        "history": [{"thought": "I will take the apple.",
                     "action": "take apple 1 from fridge 1", "observation": "Nothing happens.",
                     "before_state_hash": "closed", "state_hash": "closed"}],
        "failure": {"type": "explicit_failure", "command": "take apple 1 from fridge 1",
                    "observation": "Nothing happens."},
        "rubric": {
            "evidence": "The take command failed because the fridge is closed.",
            "blocked_subgoal": "take apple 1 from fridge 1",
            "detection_targets": ["apple", "fridge"],
            "acceptable_directions": ["Open fridge 1 before taking the apple."],
            "strategy_rules": [{"action_regex": r"^open fridge 1$", "require_state_change": True}],
            "recovery_rules": [{"action_regex": r"^take apple 1 from fridge 1$",
                                "require_state_change": True}],
        },
    }


def step(thought="I will look around.", action="look", changed=False, won=False, facts=None,
         observation="You are at fridge 1."):
    return {"thought": thought, "action": action, "observation": observation,
            "before_state_hash": "before", "state_hash": "after" if changed else "before",
            "won": won, "done": won, "facts": [] if facts is None else facts,
            "parse_ok": True}


def run(steps):
    return {"checkpoint_id": "test-error", "condition": "A", "status": "completed",
            "steps": steps, "won": any(s["won"] for s in steps)}


class ScoringProtocolTests(unittest.TestCase):
    def test_strategy_does_not_require_language_detection(self):
        scored = score_checkpoint(error_checkpoint(), run([
            step("I will open the fridge.", "open fridge 1", changed=True)
        ]))
        self.assertEqual(scored["detection_3"]["score"], 0)
        self.assertEqual(scored["strategy_5"]["score"], 1)

    def test_declared_correction_without_valid_state_change_does_not_score(self):
        scored = score_checkpoint(error_checkpoint(), run([
            step("The fridge is closed; I will open it.", "open fridge 1",
                 observation="Nothing happens.")
        ]))
        self.assertEqual(scored["strategy_5"]["score"], 0)
        self.assertEqual(scored["recovery_10"]["score"], 0)

    def test_generic_retry_is_not_specific_detection(self):
        scored = score_checkpoint(error_checkpoint(), run([
            step("That did not work; I will try again.")
        ]))
        self.assertEqual(scored["detection_3"]["score"], 0)
        self.assertTrue(scored["detection_3"]["provisional"])

    def test_specific_failed_subgoal_is_provisional_detection(self):
        scored = score_checkpoint(error_checkpoint(), run([
            step("Taking the apple failed because the fridge is closed.")
        ]))
        self.assertEqual(scored["detection_3"]["score"], 1)

    def test_plain_ordering_plan_is_not_error_detection(self):
        scored = score_checkpoint(error_checkpoint(), run([
            step("I will put the apple away before leaving.")
        ]))
        self.assertEqual(scored["detection_3"]["score"], 0)

    def test_detection_after_third_action_does_not_score(self):
        scored = score_checkpoint(error_checkpoint(), run([
            step(), step(), step(),
            step("Taking the apple failed because the fridge is closed.")
        ]))
        self.assertEqual(scored["detection_3"]["score"], 0)

    def test_strategy_after_fifth_action_does_not_score(self):
        scored = score_checkpoint(error_checkpoint(), run([
            *[step() for _ in range(5)], step(action="open fridge 1", changed=True)
        ]))
        self.assertEqual(scored["strategy_5"]["score"], 0)

    def test_recovery_at_ten_but_not_eleven(self):
        recovered = step(action="take apple 1 from fridge 1", changed=True)
        at_ten = score_checkpoint(error_checkpoint(), run([*[step() for _ in range(9)], recovered]))
        at_eleven = score_checkpoint(error_checkpoint(), run([*[step() for _ in range(10)], recovered]))
        self.assertEqual(at_ten["recovery_10"]["score"], 1)
        self.assertEqual(at_eleven["recovery_10"]["score"], 0)

    def test_required_facts_prevent_false_recovery(self):
        cp = error_checkpoint()
        cp["rubric"]["recovery_rules"][0]["required_facts"] = ["in(apple 1, inventory)"]
        missing = score_checkpoint(cp, run([
            step(action="take apple 1 from fridge 1", changed=True, facts=["at(fridge 1)"])
        ]))
        present = score_checkpoint(cp, run([
            step(action="take apple 1 from fridge 1", changed=True, facts=["in(apple 1, inventory)"])
        ]))
        self.assertEqual(missing["recovery_10"]["score"], 0)
        self.assertEqual(present["recovery_10"]["score"], 1)

    def test_success_uses_environment_won_and_twenty_action_limit(self):
        declared = score_checkpoint(error_checkpoint(), run([step("The task is now complete.")]))
        official = score_checkpoint(error_checkpoint(), run([step(won=True)]))
        too_late = score_checkpoint(error_checkpoint(), run([*[step() for _ in range(20)], step(won=True)]))
        self.assertEqual(declared["success_20"]["score"], 0)
        self.assertEqual(official["success_20"]["score"], 1)
        self.assertEqual(too_late["success_20"]["score"], 0)

    def test_normal_new_post_checkpoint_failure_is_not_false_positive(self):
        cp = error_checkpoint()
        cp["kind"] = "normal"
        cp["failure"] = None
        cp["history"][-1]["action"] = "go to fridge 1"
        cp["history"][-1]["observation"] = "You arrive at fridge 1. The fridge 1 is closed."
        cp["history"][-1]["state_hash"] = "at-fridge"
        scored = score_checkpoint(cp, run([
            step("I will take the apple.", "take apple 1 from fridge 1", observation="Nothing happens."),
            step("Taking the apple failed because the fridge is closed.", "open fridge 1", changed=True)
        ]))
        self.assertEqual(scored["false_positive_3"]["score"], 0)

    def test_report_keeps_missing_runs_in_assigned_error_denominator(self):
        first = error_checkpoint()
        second = error_checkpoint()
        second["checkpoint_id"] = "missing-error"
        second["game_id"] = "synthetic-game-two"
        completed = run([step(action="open fridge 1", changed=True)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints.jsonl").write_text("\n".join(json.dumps(c) for c in (first, second)) + "\n")
            (root / "runs.jsonl").write_text(json.dumps(completed) + "\n")
            rows = score_directory(root, bootstrap_samples=100, seed=1)
            by_metric = {row["metric"]: row for row in rows}
            strategy = by_metric["strategy_5"]
            self.assertEqual(strategy["a_numerator"], 1)
            self.assertEqual(strategy["a_denominator"], 2)
            self.assertEqual(strategy["b_numerator"], 0)
            self.assertEqual(strategy["b_denominator"], 2)
            self.assertEqual(strategy["a_minus_b_pp"], 50)
            self.assertEqual(strategy["a_only"], 1)
            self.assertEqual(strategy["b_only"], 0)
            self.assertEqual(strategy["assigned_pairs"], 2)
            self.assertEqual(strategy["completed_pairs"], 0)
            self.assertTrue((root / "report.csv").exists())
            self.assertTrue((root / "report.md").exists())
            with (root / "blind_review.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertNotIn("condition", reader.fieldnames)
                self.assertNotIn("session_id", reader.fieldnames)

    def test_bootstrap_resamples_whole_games_and_preserves_pairs(self):
        first = error_checkpoint()
        second = error_checkpoint()
        second["checkpoint_id"] = "same-game-other-checkpoint"
        # One game contributes one +1 and one -1 difference. Every game-cluster
        # bootstrap sample must retain both differences, so its mean is zero.
        records = []
        for cp, winner in ((first, "A"), (second, "B")):
            for condition in ("A", "B"):
                record = run([step(action="open fridge 1", changed=condition == winner)])
                record["checkpoint_id"] = cp["checkpoint_id"]
                record["condition"] = condition
                records.append(record)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints.jsonl").write_text("\n".join(json.dumps(c) for c in (first, second)) + "\n")
            (root / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
            rows = score_directory(root, bootstrap_samples=100, seed=5)
            strategy = next(row for row in rows if row["metric"] == "strategy_5")
            self.assertEqual(strategy["paired_games"], 1)
            self.assertEqual(strategy["a_minus_b_pp"], 0)
            self.assertEqual(strategy["ci95_low_pp"], 0)
            self.assertEqual(strategy["ci95_high_pp"], 0)
            self.assertEqual(strategy["a_only"], 1)
            self.assertEqual(strategy["b_only"], 1)


if __name__ == "__main__":
    unittest.main()
