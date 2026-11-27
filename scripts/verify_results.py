#!/usr/bin/env python3
"""Verify stored protocol evidence and replay actual ALFWorld continuations.

This makes no model/server requests and does not modify benchmark inputs or scores.
It writes protocol_audit.json atomically. Verified replays are reused only for the
same run/checkpoint/game hashes and installed environment implementation. Pass
--force-replay to independently execute every continuation again.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from react_ablation.experiment import assert_snapshot, failure_evidence, protocol_identity, replay_checkpoint
from react_ablation.model import ModelConfig
from react_ablation.prompting import SYSTEM, assert_mask_invariance, digest, parse_response, render_prompt
from react_ablation.scoring import rule_matches
from react_ablation.storage import read_jsonl, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def before(a, b):
    return datetime.fromisoformat(a) <= datetime.fromisoformat(b)


def verify_results(run_dir, allow_incomplete=False, force_replay=False):
    root = Path(run_dir).resolve()
    frozen = json.loads((root / "freeze_manifest.json").read_text())
    checkpoints = read_jsonl(root / "checkpoints.jsonl")
    candidates = {c["checkpoint_id"]: c for c in read_jsonl(root / "candidates.jsonl")}
    probes = {p["checkpoint_id"]: p for p in read_jsonl(root / "recoverability_probes.jsonl")}
    # Bind the audit to one immutable input snapshot even when a live runner
    # appends further completed sessions during --allow-incomplete verification.
    runs_path = root / "runs.jsonl"
    runs_bytes = runs_path.read_bytes() if runs_path.exists() else b""
    runs = [json.loads(line) for line in runs_bytes.splitlines() if line.strip()]
    runs_snapshot_sha = hashlib.sha256(runs_bytes).hexdigest()
    allocation = json.loads((root / "allocation.json").read_text())
    cp_by_id = {c["checkpoint_id"]: c for c in checkpoints}
    expected = sum(frozen["counts"].values())
    config = asdict(ModelConfig())
    identity = protocol_identity()
    run_keys = [(r["checkpoint_id"], r["condition"]) for r in runs]
    first_start = min((r["started_utc"] for r in runs), default=None)
    old_path = root / "protocol_audit.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    audit = {"schema_version": 2, "audited_utc": utc_now(),
             "scope": "File evidence plus actual official ALFWorld action replay; no model requests or cache operations",
             "allow_incomplete": allow_incomplete, "force_replay": force_replay,
             "expected_checkpoints": expected, "expected_sessions": 2 * expected,
             "checks": {}, "checkpoint_checks": [], "session_checks": [], "continuation_replays": []}
    checks = audit["checks"]
    checks.update(
        frozen_count_matches=len(checkpoints) == expected,
        kind_counts_match=all(sum(c["kind"] == kind for c in checkpoints) == count for kind, count in frozen["counts"].items()),
        unique_checkpoint_ids=len(cp_by_id) == len(checkpoints),
        one_checkpoint_per_game=len({c["game_id"] for c in checkpoints}) == len(checkpoints),
        checkpoints_sha_matches=sha(root / "checkpoints.jsonl") == frozen["checkpoints_sha256"],
        protocol_identity_matches_current=frozen["protocol_identity"] == identity,
        frozen_model_config_matches=frozen["model_config"] == config,
        official_placement_syntax="move OBJECT to RECEPTACLE" in SYSTEM and "put OBJECT in/on" not in SYSTEM,
        allocation_ids_exact=set(allocation) == set(cp_by_id),
        allocation_has_A_B_once_each=all(sorted(a["order"]) == ["A", "B"] for a in allocation.values()),
        runs_unique=len(run_keys) == len(set(run_keys)),
        runs_all_completed=all(r["status"] == "completed" for r in runs),
        runs_have_valid_checkpoint_condition=all(cid in cp_by_id and cond in ("A", "B") for cid, cond in run_keys),
        frozen_before_first_run=first_start is None or before(frozen["frozen_utc"], first_start),
    )
    if not allow_incomplete:
        checks["all_assigned_sessions_completed"] = set(run_keys) == {(c["checkpoint_id"], cond) for c in checkpoints for cond in ("A", "B")}
    checks["actual_condition_order_matches_allocation"] = all(
        [r["condition"] for r in sorted((r for r in runs if r["checkpoint_id"] == cid), key=lambda r: datetime.fromisoformat(r["started_utc"]))]
        == [condition for condition in assigned["order"] if (cid, condition) in set(run_keys)]
        for cid, assigned in allocation.items())
    chronological = sorted(runs, key=lambda r: datetime.fromisoformat(r["started_utc"]))
    checks["independent_sessions_do_not_overlap"] = all(before(left["ended_utc"], right["started_utc"])
        for left, right in zip(chronological, chronological[1:]))
    preflight_status = root / "preflight_old_placement_syntax/status.json"
    if preflight_status.exists():
        checks["archived_preflight_has_no_benchmark_pairs"] = json.loads(preflight_status.read_text())["benchmark_pairs_executed"] == 0
    for cp in checkpoints:
        cid, history = cp["checkpoint_id"], cp["history"]
        mask = assert_mask_invariance(cp["initial"], history)
        c = {"checkpoint_id": cid, "kind": cp["kind"], "history_depth": len(history), "checks": {}}
        cc = c["checks"]
        cc.update(
            final_candidate_history_unchanged=cid in candidates and history == candidates[cid]["history"] and cp["initial"] == candidates[cid]["initial"],
            collection_prompt_hashes_match=all(digest(render_prompt(cp["initial"], history[:i], "A", 0)[0]) == row["model"]["prompt_sha256"] for i, row in enumerate(history)),
            collection_config_matches=all(row["model"]["config"] == config for row in history),
            collection_untruncated=all(not row["model"]["truncated"] for row in history),
            collection_first_cache_zero=history[0]["model"]["cached_tokens"] == 0,
            mask_hashes_match=all(mask[k] == cp["mask_validation"][k] for k in ("prompt_a_sha256", "prompt_b_sha256")),
            replay_preregistered=cp["replay_validation"]["passed"] and before(cp["replay_validation"]["verified_utc"], frozen["frozen_utc"]),
            rubric_frozen_before_runs=first_start is None or before(cp["rubric_frozen_utc"], first_start),
            first_eligible_boundary=all(failure_evidence(history[:i]) is None for i in range(1, len(history))),
        )
        if cp["kind"] == "error":
            p = probes.get(cid)
            cc["independent_probe_exists"] = p is not None
            if p:
                cc.update(
                    probe_precedes_freeze=before(p["verified_utc"], cp["rubric_frozen_utc"]),
                    probe_starts_at_checkpoint=p["initial_checkpoint_state"]["state_hash"] == history[-1]["state_hash"],
                    probe_actions_preregistered=[s["action"] for s in p["steps"]] == cp["rubric"]["probe_actions"],
                    recovery_probe_matches_frozen_rules=p["recovery_verified"] and any(rule_matches(rule, s) for rule in cp["rubric"]["recovery_rules"] for s in p["steps"]),
                    strategy_probe_matches_frozen_rules=p["strategy_verified"] and any(rule_matches(rule, s) for rule in cp["rubric"]["strategy_rules"] for s in p["steps"]),
                    oracle_actions_excluded=cp["rubric"]["oracle_actions_excluded_from_model_history"],
                )
            cc["checkpoint_has_error_evidence"] = failure_evidence(history) is not None
        else:
            cc["normal_has_no_prior_error"] = failure_evidence(history) is None
        audit["checkpoint_checks"].append(c)

    for run in runs:
        cid, condition = run["checkpoint_id"], run["condition"]
        cp, assigned = cp_by_id[cid], allocation[cid]
        history, steps = cp["history"], run["steps"]
        prompt_ok, old_mask_ok, new_thought_ok = [], [], []
        for i, row in enumerate(steps):
            prompt, spans = render_prompt(cp["initial"], history + steps[:i], condition, len(history))
            prompt_ok.append(digest(prompt) == row["model"]["prompt_sha256"])
            if condition == "B":
                old_mask_ok.extend(prompt[a:b] == "[MASKED]" for a, b in spans[:len(history)])
                new_thought_ok.extend(prompt[a:b] == prior["thought"] for (a, b), prior in zip(spans[len(history):], steps[:i]))
        parsed = [parse_response(s["raw_response"]) for s in steps]
        session_path = root / "sessions" / f"{cid}_{condition}.json"
        sc = {
            "budget_at_most20": len(steps) <= 20,
            "termination_valid": len(steps) == 20 or bool(steps and steps[-1]["done"]),
            "model_config_matches": run["model_config"] == config and all(s["model"]["config"] == config for s in steps),
            "model_protocol_matches": run["protocol_identity"] == frozen["protocol_identity"],
            "started_after_freeze": before(frozen["frozen_utc"], run["started_utc"]),
            "independent_replay_marked": run["independent_environment_replay"],
            "cache_erasure_confirmed": "n_erased" in run["cache_erasure"],
            "first_cache_zero": bool(steps) and steps[0]["model"]["cached_tokens"] == 0,
            "all_prompt_hashes_match": all(prompt_ok),
            "old_B_thoughts_masked": all(old_mask_ok),
            "new_B_thoughts_retained": all(new_thought_ok),
            "all_untruncated": all(not s["model"]["truncated"] for s in steps),
            "parsed_response_matches_logged_action": all(p["parse_ok"] and p["thought"] == s["thought"] and p["action"] == s["action"] for p, s in zip(parsed, steps)),
            "paired_seed_offsets_match": run["seed"] == assigned["seed"] and all(s["model"]["seed"] == assigned["seed"] + i for i, s in enumerate(steps)),
            "allocated_order_matches": run["order"] == assigned["order"].index(condition) + 1,
            "matches_durable_session": session_path.exists() and run == json.loads(session_path.read_text()),
        }
        audit["session_checks"].append({"checkpoint_id": cid, "condition": condition, "status": run["status"], "steps": len(steps), "checks": sc})

    # Replay only structurally valid evidence, without making inference requests.
    preliminary = [checks] + [r["checks"] for r in audit["checkpoint_checks"] + audit["session_checks"]]
    if all(all(c.values()) for c in preliminary):
        from react_ablation.environment import resolve_game
        environment_identity = {"implementation_sha256": sha(PROJECT / "src/react_ablation/environment.py"),
                                "packages": {name: version(name) for name in ("alfworld", "textworld", "numpy")}}
        audit["replay_environment_identity"] = environment_identity
        prior = {r["session_id"]: r for r in old.get("continuation_replays", [])} if old.get("replay_environment_identity") == environment_identity and not force_replay else {}
        for run in runs:
            cp = cp_by_id[run["checkpoint_id"]]
            actual_game_sha = sha(resolve_game(cp["game_file"]))
            if actual_game_sha != cp["game_sha256"]:
                raise RuntimeError(f"Game data changed: {cp['game_id']}")
            cached = prior.get(run["session_id"])
            if cached and cached["run_sha256"] == digest(run) and cached.get("checkpoints_sha256") == frozen["checkpoints_sha256"] and cached.get("game_sha256") == actual_game_sha:
                audit["continuation_replays"].append(cached)
                continue
            env = replay_checkpoint(cp)
            try:
                for row in run["steps"]:
                    assert_snapshot(env.step(row["action"]), row)
                audit["continuation_replays"].append({"session_id": run["session_id"], "checkpoint_id": run["checkpoint_id"], "condition": run["condition"],
                    "run_sha256": digest(run), "checkpoints_sha256": frozen["checkpoints_sha256"], "game_sha256": actual_game_sha,
                    "checkpoint_actions_verified": len(cp["history"]), "continuation_actions_verified": len(run["steps"]),
                    "fields": ["observation", "state_hash", "admissible_commands", "won", "done"], "passed": True, "verified_utc": utc_now()})
            finally:
                env.close()
            # Checkpoint audit progress durably in case verification is interrupted.
            audit["verification_in_progress"] = True
            write_json(root / "protocol_audit.json", audit)
            print(json.dumps({"verified_session": run["session_id"], "condition": run["condition"], "verified_count": len(audit["continuation_replays"])}), flush=True)

    checks["every_completed_session_actually_replayed"] = len(audit["continuation_replays"]) == len(runs)
    audit["completed_pairs"] = sum(all((cp["checkpoint_id"], c) in set(run_keys) for c in ("A", "B")) for cp in checkpoints)
    audit["completed_session_count"] = len(runs)
    audit["continuation_replay_actions_verified"] = sum(r["continuation_actions_verified"] for r in audit["continuation_replays"])
    audit["all_expected_sessions_completed"] = len(runs) == 2 * expected
    audit["final_all_24_sessions"] = expected == 12 and len(runs) == 24  # Pilot compatibility key.
    audit["all_assigned_sessions_verified"] = len(audit["continuation_replays"]) == 2 * expected
    audit["all_checks_passed"] = all(all(c.values()) for c in [checks] + [r["checks"] for r in audit["checkpoint_checks"] + audit["session_checks"]])
    audit["verification_in_progress"] = False
    audit["audited_utc"] = utc_now()
    audit["file_hashes"] = {name: sha(root / name) for name in ("checkpoints.jsonl", "freeze_manifest.json", "allocation.json")}
    audit["file_hashes"]["runs.jsonl"] = runs_snapshot_sha
    audit["live_runs_sha256_at_audit_end"] = sha(runs_path) if runs_path.exists() else hashlib.sha256(b"").hexdigest()
    audit["runs_appended_during_audit"] = audit["live_runs_sha256_at_audit_end"] != runs_snapshot_sha
    write_json(root / "protocol_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit only completed conditions while a pilot is still running")
    parser.add_argument("--force-replay", action="store_true", help="Do not reuse matching earlier actual replay evidence")
    args = parser.parse_args()
    result = verify_results(**vars(args))
    print(json.dumps({"passed": result["all_checks_passed"], "completed_pairs": result["completed_pairs"],
                      "completed_sessions": result["completed_session_count"], "replayed_actions": result["continuation_replay_actions_verified"]}), flush=True)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
