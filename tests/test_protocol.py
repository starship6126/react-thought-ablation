"""Protocol regression fixtures are synthetic and are never benchmark outputs."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from react_ablation.experiment import collect, failure_evidence
from react_ablation.prompting import assert_mask_invariance, digest, render_prompt
from react_ablation.storage import append_jsonl, read_jsonl, write_json


def snapshot(observation="A quiet room.", state_hash="s0", **extra):
    return {"observation": observation, "state_hash": state_hash,
            "won": False, "done": False, "admissible_commands": ["look"], **extra}


def step(action="look", thought="I will inspect the room.", **extra):
    return snapshot(action=action, thought=thought, before_state_hash="s0", **extra)


class PromptTests(unittest.TestCase):
    def test_only_precheckpoint_thought_bodies_are_masked(self):
        initial = snapshot("Task: put the apple on the table. Keep Thought: literal text.")
        history = [step(thought="Plan with café and Thought: marker.", action="go to countertop 1",
                        observation="Observation includes Thought: and [MASKED] literals."),
                   step(thought="I will take apple 1.", action="take apple 1 from countertop 1"),
                   step(thought="A newly generated plan must remain visible.", action="go to diningtable 1")]
        a, spans = render_prompt(initial, history, "A", 2)
        b, _ = render_prompt(initial, history, "B", 2)
        expected = a
        for begin, end in reversed(spans[:2]):
            expected = expected[:begin] + "[MASKED]" + expected[end:]
        self.assertEqual(expected, b)
        self.assertIn(history[2]["thought"], b)
        self.assertIn(initial["observation"], b)
        for row in history:
            self.assertIn("Action: " + row["action"], b)
            self.assertIn("Observation: " + row["observation"], b)
        self.assertTrue(assert_mask_invariance(initial, history[:2])["passed"])


class EligibilityTests(unittest.TestCase):
    def test_explicit_failure_captures_exact_command(self):
        evidence = failure_evidence([step(action="take apple 1", observation="Nothing happens.")])
        self.assertEqual("explicit_failure", evidence["type"])
        self.assertEqual("take apple 1", evidence["command"])

    def test_unchanged_loop_is_eligible(self):
        rows = [step(action="look"), step(action="inventory"), step(action="look")]
        self.assertEqual("unchanged_repeated_command", failure_evidence(rows)["type"])

    def test_new_intervening_clue_excludes_loop(self):
        rows = [step(action="look"), step(action="examine apple 1", observation="The apple is dirty."),
                step(action="look")]
        self.assertIsNone(failure_evidence(rows))

    def test_changed_state_excludes_loop(self):
        rows = [step(action="look"), step(action="look", state_hash="s1")]
        self.assertIsNone(failure_evidence(rows))


class CollectionResumeTests(unittest.TestCase):
    def fake_environment(self, game, initial, saved_steps):
        calls = []

        class Session:
            def __init__(self, path, seed):
                self.index = 0
                calls.append(("init", str(path), seed))

            def reset(self):
                return initial.copy()

            def step(self, action):
                expected = saved_steps[self.index]
                self.index += 1
                calls.append(("step", action))
                if action != expected["action"]:
                    raise AssertionError("Replay action changed")
                return expected.copy()

            def close(self):
                calls.append(("close",))

        module = types.ModuleType("react_ablation.environment")
        module.AlfworldSession = Session
        module.list_games = lambda split: [game]
        module.game_id = lambda path: "seen/game1"
        return module, calls

    def test_saved_failure_is_committed_without_new_model_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / "game.tw-pddl"
            game.write_text("synthetic test fixture")
            initial = snapshot()
            failed = step(action="take apple 1", observation="Nothing happens.")
            module, calls = self.fake_environment(game, initial, [failed])
            write_json(root / "collection_sessions" / (digest("seen/game1")[:16] + ".json"),
                       {"initial": initial, "history": [failed]})
            with patch.dict(sys.modules, {"react_ablation.environment": module}), \
                 patch("react_ablation.experiment.LocalModel"), \
                 patch("react_ablation.experiment.take_step", side_effect=AssertionError("Must not generate past boundary")):
                selected = collect(root, errors=1, normals=0, max_steps=5)
            self.assertEqual(1, len(selected))
            self.assertEqual([failed], selected[0]["history"])
            self.assertEqual(1, len([call for call in calls if call[0] == "step"]))

    def test_existing_candidate_recovers_ledger_without_reassigning_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / "game.tw-pddl"
            game.write_text("synthetic test fixture")
            candidate = {"checkpoint_id": "already-saved", "game_id": "seen/game1", "seed": 20261127,
                         "kind": "normal", "history": [step(state_hash="s1")]}
            append_jsonl(root / "candidates.jsonl", candidate)
            module, calls = self.fake_environment(game, snapshot(), [])
            with patch.dict(sys.modules, {"react_ablation.environment": module}), \
                 patch("react_ablation.experiment.LocalModel"), \
                 patch("react_ablation.experiment.take_step", side_effect=AssertionError("Duplicate game inference")):
                selected = collect(root, errors=1, normals=1, max_steps=5)
            self.assertEqual([candidate], selected)
            self.assertEqual([], calls)
            ledger = read_jsonl(root / "collection_games.jsonl")
            self.assertTrue(ledger[0]["recovered_existing_candidate"])
            self.assertEqual(1, len(read_jsonl(root / "candidates.jsonl")))

    def test_saved_failure_in_normal_game_stops_without_later_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / "game.tw-pddl"
            game.write_text("synthetic test fixture")
            initial = snapshot()
            failed = step(action="take apple 1", observation="Nothing happens.")
            module, calls = self.fake_environment(game, initial, [failed])
            write_json(root / "collection_sessions" / (digest("seen/game1")[:16] + ".json"),
                       {"initial": initial, "history": [failed], "normal_candidate_designated": True})
            with patch.dict(sys.modules, {"react_ablation.environment": module}), \
                 patch("react_ablation.experiment.LocalModel"), \
                 patch("react_ablation.experiment.take_step", side_effect=AssertionError("Must stop failed normal game")):
                selected = collect(root, errors=0, normals=1, max_steps=5)
            self.assertEqual([], selected)
            self.assertIsNone(read_jsonl(root / "collection_games.jsonl")[0]["candidate"])

    def test_changed_initial_state_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / "game.tw-pddl"
            game.write_text("synthetic test fixture")
            module, calls = self.fake_environment(game, snapshot(state_hash="changed"), [])
            write_json(root / "collection_sessions" / (digest("seen/game1")[:16] + ".json"),
                       {"initial": snapshot(), "history": []})
            with patch.dict(sys.modules, {"react_ablation.environment": module}), \
                 patch("react_ablation.experiment.LocalModel"), \
                 self.assertRaisesRegex(AssertionError, "state_hash"):
                collect(root, errors=1, normals=0, max_steps=5)


if __name__ == "__main__":
    unittest.main()
