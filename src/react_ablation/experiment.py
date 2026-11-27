"""Collection, frozen eligibility, independent replay and durable paired execution."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import uuid

from .model import LocalModel, ModelConfig
from .prompting import render_prompt, assert_mask_invariance, digest
from .storage import read_jsonl, write_json, append_jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol_identity():
    from .prompting import SYSTEM
    from .model import GRAMMAR
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((repo / "artifacts/setup/runtime_manifest.json").read_text())
    return {"runtime": manifest["identity"], "system_prompt_sha256": digest(SYSTEM),
            "grammar_sha256": digest(GRAMMAR), "decoding": asdict(ModelConfig())}


def failure_evidence(history):
    if not history:
        return None
    last = history[-1]
    if re.search(r"nothing happens|can't|cannot|don't understand|not a valid|invalid command|you have to", last["observation"], re.I):
        return {"type": "explicit_failure", "command": last["action"],
                "observation": last["observation"], "at_step": len(history)}
    recent = history[-3:]
    equal = [s for s in recent if s["action"] == last["action"]]
    # A different intervening observation can contain a newly discovered clue even
    # when the physical state does not change. Conservatively exclude that window.
    if len(equal) >= 2 and len({s["state_hash"] for s in recent}) == 1 and len({s["observation"] for s in recent}) == 1:
        return {"type": "unchanged_repeated_command", "command": last["action"],
                "count_in_last_three": len(equal), "observation": last["observation"],
                "at_step": len(history), "unchanged_state_hash": last["state_hash"]}
    return None


def take_step(env, model, initial, history, condition, checkpoint_length, seed):
    prompt, _ = render_prompt(initial, history, condition, checkpoint_length)
    generated = model.generate(prompt, seed)
    if not generated["parse_ok"]:
        raise RuntimeError("Model response parsing failed: " + json.dumps(generated))
    before = history[-1] if history else initial
    snapshot = env.step(generated["action"])
    # Initial snapshots retain the entity dictionary; each transition retains
    # complete readable facts/hash without repeating redundant structured facts.
    snapshot = {k: v for k, v in snapshot.items() if k not in ("fact_records", "entities", "raw_fact_records")}
    return {**snapshot, **generated, "before_state_hash": before["state_hash"],
            "admissible_before": before["admissible_commands"], "recorded_utc": now()}


def make_checkpoint(game, split, seed, initial, history, kind, evidence):
    from .environment import game_id
    identity = game_id(game)
    cp_id = hashlib.sha256(f"{identity}|{seed}|{len(history)}|{kind}".encode()).hexdigest()[:16]
    return {"checkpoint_id": cp_id, "game_id": identity, "game_file": str(game),
            "game_sha256": hashlib.sha256(Path(game).read_bytes()).hexdigest(),
            "split": split, "seed": seed, "kind": kind, "initial": initial,
            "history": history.copy(), "failure": evidence, "created_utc": now(),
            "checkpoint_boundary": "after observation, before next model thought",
            "mask_validation": assert_mask_invariance(initial, history)}


def collect(run_dir, split="seen", errors=8, normals=4, max_steps=25, seed=20261127, all_games=False):
    from .environment import AlfworldSession, list_games, game_id
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_file = root / "collection_config.json"
    cfg = {"split": split, "errors_target": errors, "normals_target": normals,
           "max_steps": max_steps, "seed": seed, "all_games": all_games,
           "model_config": asdict(ModelConfig()), "selection": "first eligible error per game; disjoint normal games"}
    if config_file.exists() and json.loads(config_file.read_text()) != cfg:
        raise RuntimeError("Cannot change a collection configuration when resuming")
    write_json(config_file, cfg)
    games = list_games(split)
    random.Random(seed).shuffle(games)
    if split == "unseen" and len(games) != 134:
        raise RuntimeError(f"Expected official 134 unseen games, got {len(games)}")
    write_json(root / "game_order.json", [{"game_id": game_id(g), "game_file": str(g)} for g in games])
    selected = read_jsonl(root / "candidates.jsonl")
    selected_by_game = {cp["game_id"]: cp for cp in selected}
    if len(selected_by_game) != len(selected):
        raise RuntimeError("A game has multiple candidate checkpoints")
    completed = {row["game_id"] for row in read_jsonl(root / "collection_games.jsonl")}
    model = LocalModel()
    for game_index, game in enumerate(games):
        identity = game_id(game)
        if identity in completed:
            continue
        if identity in selected_by_game:
            # Recover an interruption between candidate commit and game-ledger commit.
            cp = selected_by_game[identity]
            append_jsonl(root / "collection_games.jsonl", {"game_id": identity, "seed": cp["seed"],
                          "steps": len(cp["history"]), "candidate": cp["checkpoint_id"],
                          "normal_candidate_designated": cp["kind"] == "normal", "ended_utc": now(),
                          "recovered_existing_candidate": True})
            completed.add(identity)
            continue
        n_error = sum(c["kind"] == "error" for c in selected)
        n_normal = sum(c["kind"] == "normal" for c in selected)
        if n_error >= errors and n_normal >= normals and not all_games:
            break
        # Normal games are selected in advance; their first clean state-changing step is retained.
        want_normal = n_normal < normals
        game_seed = seed + game_index
        journal = root / "collection_sessions" / (digest(identity)[:16] + ".json")
        old = json.loads(journal.read_text()) if journal.exists() else None
        if old is not None:
            if old.get("seed", game_seed) != game_seed or old.get("game_id", identity) != identity:
                raise RuntimeError("Collection journal game or seed changed")
            want_normal = old.get("normal_candidate_designated", want_normal)

        def boundary(history):
            """Return the first eligible boundary without generating a later thought."""
            row = history[-1]
            evidence = failure_evidence(history)
            if row["done"]:
                return True, None, None
            if want_normal:
                if evidence:
                    return True, None, None
                if row["state_hash"] != row["before_state_hash"]:
                    return True, "normal", None
            elif evidence:
                return True, "error", evidence
            return False, None, None

        env = AlfworldSession(game, game_seed)
        try:
            model.begin_session()
            initial = env.reset()
            history = []
            found = None
            stop = False
            if old is not None:
                assert_snapshot(initial, old["initial"])
                for old_step in old["history"]:
                    replayed = env.step(old_step["action"])
                    assert_snapshot(replayed, old_step)
                    history.append(old_step)
                    stop, kind, evidence = boundary(history)
                    if stop:
                        if kind:
                            found = make_checkpoint(game, split, game_seed, initial, history, kind, evidence)
                        break
            # Record the game designation before any inference so resume cannot
            # silently switch the intended normal/error collection role.
            write_json(journal, {"game_id": identity, "seed": game_seed,
                       "normal_candidate_designated": want_normal, "initial": initial, "history": history})
            for step_index in range(len(history), max_steps) if not stop else ():
                row = take_step(env, model, initial, history, "A", 0, game_seed + 1000 + step_index)
                history.append(row)
                write_json(journal, {"game_id": identity, "seed": game_seed,
                           "normal_candidate_designated": want_normal, "initial": initial, "history": history})
                evidence = failure_evidence(history)
                print(json.dumps({"phase": "collect", "game": game_index + 1, "step": len(history),
                                  "action": row["action"], "observation": row["observation"],
                                  "thought": row["thought"], "failure": evidence}), flush=True)
                stop, kind, evidence = boundary(history)
                if stop:
                    if kind:
                        found = make_checkpoint(game, split, game_seed, initial, history, kind, evidence)
                    break
            if found and not any(c["checkpoint_id"] == found["checkpoint_id"] for c in selected):
                append_jsonl(root / "candidates.jsonl", found)
                selected.append(found)
                selected_by_game[identity] = found
            append_jsonl(root / "collection_games.jsonl", {"game_id": identity, "seed": game_seed,
                          "steps": len(history), "candidate": found["checkpoint_id"] if found else None,
                          "normal_candidate_designated": want_normal, "ended_utc": now()})
        finally:
            env.close()
    return selected


def assert_snapshot(actual, expected):
    for key in ("observation", "state_hash", "won", "done", "admissible_commands"):
        if actual[key] != expected[key]:
            raise AssertionError(f"Environment replay mismatch in {key}")


def replay_checkpoint(checkpoint):
    from .environment import AlfworldSession, resolve_game
    cp = checkpoint
    game = resolve_game(cp["game_file"])
    if hashlib.sha256(game.read_bytes()).hexdigest() != cp["game_sha256"]:
        raise RuntimeError("Game file changed")
    env = AlfworldSession(game, cp["seed"])
    try:
        assert_snapshot(env.reset(), cp["initial"])
        for step in cp["history"]:
            assert_snapshot(env.step(step["action"]), step)
        return env
    except Exception:
        env.close()
        raise


def freeze(run_dir, rubrics_file):
    root = Path(run_dir)
    if (root / "checkpoints.jsonl").exists():
        raise RuntimeError("Checkpoints already frozen; use a new run ID to change selection")
    candidates = read_jsonl(root / "candidates.jsonl")
    rubrics = json.loads(Path(rubrics_file).read_text())
    frozen = []
    for cp in candidates:
        if cp["checkpoint_id"] not in rubrics:
            continue
        rubric = rubrics[cp["checkpoint_id"]]
        if cp["kind"] == "error":
            for key in ("evidence", "blocked_subgoal", "detection_targets", "acceptable_directions", "strategy_rules", "recovery_rules"):
                if not rubric.get(key):
                    raise ValueError(f"Missing preregistered {key} for {cp['checkpoint_id']}")
            if not rubric.get("recoverability_verified"):
                raise ValueError("Recovery must be verified in a separate environment before A/B")
        env = replay_checkpoint(cp)
        env.close()
        cp["replay_validation"] = {"passed": True, "steps": len(cp["history"]), "verified_utc": now(),
                                  "fields": ["observation", "state_hash", "won", "done", "admissible_commands"]}
        cp["rubric"] = rubric
        cp["rubric_frozen_utc"] = now()
        frozen.append(cp)
    if not frozen:
        raise ValueError("No reviewed checkpoints")
    config = json.loads((root / "collection_config.json").read_text())
    counts = {kind: sum(cp["kind"] == kind for cp in frozen) for kind in ("error", "normal")}
    if counts != {"error": config["errors_target"], "normal": config["normals_target"]}:
        raise ValueError(f"Target counts not met: {counts}")
    # No partial freeze is published; atomic whole-file replace.
    path = root / "checkpoints.jsonl"
    temp = path.with_suffix(".partial")
    temp.write_text("")
    for cp in frozen:
        append_jsonl(temp, cp)
    temp.replace(path)
    write_json(root / "freeze_manifest.json", {"frozen_utc": now(), "counts": counts,
               "checkpoints_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "model_config": asdict(ModelConfig()), "protocol_identity": protocol_identity()})
    return frozen


def run_pairs(run_dir, seed=20261127, limit=20):
    if limit != 20:
        raise ValueError("Registered experiments use exactly a 20-action maximum")
    root = Path(run_dir)
    checkpoints = read_jsonl(root / "checkpoints.jsonl")
    freeze_manifest = json.loads((root / "freeze_manifest.json").read_text())
    if hashlib.sha256((root / "checkpoints.jsonl").read_bytes()).hexdigest() != freeze_manifest["checkpoints_sha256"]:
        raise RuntimeError("Frozen checkpoints changed")
    if freeze_manifest["model_config"] != asdict(ModelConfig()):
        raise RuntimeError("Model decoding config changed")
    if freeze_manifest["protocol_identity"] != protocol_identity():
        raise RuntimeError("Frozen model/runtime/prompt protocol changed")
    order_file = root / "allocation.json"
    if not order_file.exists():
        rng = random.Random(seed)
        allocation = {}
        for cp in checkpoints:
            order = ["A", "B"]
            rng.shuffle(order)
            allocation[cp["checkpoint_id"]] = {"order": order, "seed": rng.randrange(1, 2**30)}
        write_json(order_file, allocation)
    allocation = json.loads(order_file.read_text())
    model = LocalModel()
    existing = read_jsonl(root / "runs.jsonl")
    if any(r["status"] != "completed" for r in existing):
        raise RuntimeError("Only completed sessions belong in runs.jsonl; resume sessions journals")
    finished = {(r["checkpoint_id"], r["condition"]) for r in existing}
    if len(finished) != len(existing):
        raise RuntimeError("Duplicate condition sessions in runs.jsonl")
    for cp in checkpoints:
        info = allocation[cp["checkpoint_id"]]
        for order_index, condition in enumerate(info["order"]):
            key = (cp["checkpoint_id"], condition)
            if key in finished:
                continue
            journal = root / "sessions" / f"{key[0]}_{condition}.json"
            env = replay_checkpoint(cp)
            try:
                cache_erasure = model.begin_session()
                run = {"checkpoint_id": key[0], "condition": condition, "session_id": str(uuid.uuid4()),
                       "order": order_index + 1, "seed": info["seed"], "steps": [], "status": "running",
                       "started_utc": now(), "model_config": asdict(ModelConfig()), "won": False,
                       "protocol_identity": freeze_manifest["protocol_identity"],
                       "independent_environment_replay": True, "cache_prompt": True,
                       "cache_erasure": cache_erasure}
                if journal.exists():
                    run = json.loads(journal.read_text())
                    for previous in run["steps"]:
                        assert_snapshot(env.step(previous["action"]), previous)
                for index in range(len(run["steps"]), limit):
                    if run["steps"] and run["steps"][-1]["done"]:
                        break
                    history = cp["history"] + run["steps"]
                    row = take_step(env, model, cp["initial"], history, condition, len(cp["history"]), info["seed"] + index)
                    run["steps"].append(row)
                    run["won"] = bool(row["won"])
                    write_json(journal, run)
                    print(json.dumps({"phase": "resume", "checkpoint": key[0], "condition": condition,
                          "step": index + 1, "action": row["action"], "won": row["won"],
                          "latency": row["model"]["latency_seconds"]}), flush=True)
                    if row["done"]:
                        break
                run["status"] = "completed"
                run["ended_utc"] = now()
                write_json(journal, run)
                append_jsonl(root / "runs.jsonl", run)
                finished.add(key)
            finally:
                env.close()
        print(json.dumps({"phase": "pair_complete", "checkpoint": cp["checkpoint_id"],
                          "completed_pairs": sum(all((p["checkpoint_id"], c) in finished for c in ("A", "B")) for p in checkpoints)}), flush=True)
        from .scoring import score_directory
        score_directory(root)
    return len(finished) // 2
