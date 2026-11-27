"""Protocol regression fixtures are synthetic and are never benchmark outputs."""
import json
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from react_ablation.experiment import collect, failure_evidence, run_pairs
from react_ablation.model import ModelConfig
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


class PairedResumeTests(unittest.TestCase):
    """Only synthetic fixtures: assert no hidden extra actions after restart."""

    identity = {"runtime": {"llama_commit": "test-only-commit"}, "prompt_module_sha256": "test-only-template"}

    def prepare(self, root, saved_count=7):
        cp = {"checkpoint_id": "test-checkpoint", "game_id": "test/game", "kind": "normal",
              "initial": snapshot(), "history": [step(action="go to table 1", state_hash="checkpoint-state")]}
        append_jsonl(root / "checkpoints.jsonl", cp)
        write_json(root / "freeze_manifest.json", {
            "checkpoints_sha256": hashlib.sha256((root / "checkpoints.jsonl").read_bytes()).hexdigest(),
            "model_config": asdict(ModelConfig()), "protocol_identity": self.identity})
        write_json(root / "allocation.json", {"test-checkpoint": {"order": ["A", "B"], "seed": 333}})

        def row(index):
            return step(action=f"test action {index}", state_hash=f"state-{index}",
                        model={"latency_seconds": 0, "seed": 333 + index})

        saved = {"checkpoint_id": "test-checkpoint", "condition": "A", "session_id": "preserved-session",
                 "order": 1, "seed": 333, "steps": [row(i) for i in range(saved_count)],
                 "status": "running", "won": False, "model_config": asdict(ModelConfig()),
                 "protocol_identity": self.identity, "cache_erasure": {"n_erased": 0}}
        write_json(root / "sessions/test-checkpoint_A.json", saved)
        other = {**saved, "condition": "B", "order": 2, "session_id": "already-completed",
                 "steps": [row(i) for i in range(20)], "status": "completed"}
        append_jsonl(root / "runs.jsonl", other)
        return cp, saved, row

    def mocks(self, saved, row, generated):
        stack = ExitStack()
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch("react_ablation.experiment.protocol_identity", return_value=self.identity))
        stack.enter_context(patch("react_ablation.experiment.verify_runtime", return_value={"test_only": True}))
        model = stack.enter_context(patch("react_ablation.experiment.LocalModel")).return_value
        model.begin_session.return_value = {"id_slot": 0, "n_erased": 50}
        environment = stack.enter_context(patch("react_ablation.experiment.replay_checkpoint")).return_value
        environment.step.side_effect = [dict(s) for s in saved["steps"]]
        stack.enter_context(patch("react_ablation.scoring.score_directory"))

        def generate(env, local_model, initial, history, condition, checkpoint_length, seed):
            index = len(history) - checkpoint_length
            generated.append({"index": index, "condition": condition, "seed": seed})
            return row(index)

        stack.enter_context(patch("react_ablation.experiment.take_step", side_effect=generate))
        return stack, model, environment

    def test_resume_counts_saved_actions_in_twenty_action_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, saved, row = self.prepare(root, saved_count=7)
            generated = []
            stack, model, environment = self.mocks(saved, row, generated)
            with stack:
                self.assertEqual(1, run_pairs(root))
            self.assertEqual(list(range(7, 20)), [s["index"] for s in generated])
            self.assertEqual(list(range(340, 353)), [s["seed"] for s in generated])
            self.assertEqual({"A"}, {s["condition"] for s in generated})
            self.assertEqual(7, environment.step.call_count)
            model.begin_session.assert_called_once()
            rows = read_jsonl(root / "runs.jsonl")
            a = next(r for r in rows if r["condition"] == "A")
            self.assertEqual("preserved-session", a["session_id"])
            self.assertEqual(20, len(a["steps"]))
            self.assertEqual("completed", a["status"])

    def test_twenty_saved_actions_commit_without_extra_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, saved, row = self.prepare(root, saved_count=20)
            generated = []
            stack, model, environment = self.mocks(saved, row, generated)
            with stack:
                self.assertEqual(1, run_pairs(root))
                self.assertEqual(1, run_pairs(root))
            self.assertEqual([], generated)
            self.assertEqual(2, len(read_jsonl(root / "runs.jsonl")))
            self.assertEqual(20, environment.step.call_count)

    def test_mismatched_journal_identity_is_rejected_before_action_replay(self):
        replacements = {"checkpoint_id": "other-cp", "condition": "B", "order": 2,
                        "seed": 999, "model_config": {}, "protocol_identity": {}}
        for field, replacement in replacements.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cp, saved, row = self.prepare(root)
                saved[field] = replacement
                write_json(root / "sessions/test-checkpoint_A.json", saved)
                generated = []
                stack, model, environment = self.mocks(saved, row, generated)
                with stack, self.assertRaisesRegex(RuntimeError, "Session journal changed " + field):
                    run_pairs(root)
                environment.step.assert_not_called()
                self.assertEqual([], generated)

    def test_oversized_journal_is_rejected_before_action_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, saved, row = self.prepare(root, saved_count=21)
            generated = []
            stack, model, environment = self.mocks(saved, row, generated)
            with stack, self.assertRaisesRegex(RuntimeError, "action budget"):
                run_pairs(root)
            environment.step.assert_not_called()
            self.assertEqual([], generated)

    def test_incomplete_run_log_cannot_be_treated_as_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, saved, row = self.prepare(root)
            append_jsonl(root / "runs.jsonl", saved)
            stack, model, environment = self.mocks(saved, row, [])
            with stack, self.assertRaisesRegex(RuntimeError, "Only completed sessions"):
                run_pairs(root)
            environment.step.assert_not_called()

    def test_duplicate_completed_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, saved, row = self.prepare(root)
            append_jsonl(root / "runs.jsonl", read_jsonl(root / "runs.jsonl")[0])
            stack, model, environment = self.mocks(saved, row, [])
            with stack, self.assertRaisesRegex(RuntimeError, "Duplicate condition sessions"):
                run_pairs(root)
            environment.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
