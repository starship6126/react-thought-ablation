"""Freeze cause-specific pilot rubrics before A/B; probe separate real environments.

This is a pilot annotation script, not a policy for the experiment model. Symbolic
facts are used only by the evaluator. No recovery probe enters the model history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from react_ablation.experiment import replay_checkpoint
from react_ablation.scoring import rule_matches
from react_ablation.storage import read_jsonl, write_json

def exact(action, **kwargs):
    return {"action_regex": re.escape(action), "require_state_change": True, **kwargs}


def pairs(snapshot, predicate):
    return [tuple(arg["name"].strip() for arg in record["arguments"])
            for record in snapshot["fact_records"] if record["predicate"] == predicate]


def compact(snapshot):
    return {key: snapshot[key] for key in ("observation", "state_hash", "facts", "admissible_commands", "won", "done")}


def pick_up_plan(snapshot, target):
    sources = sorted((obj, source) for obj, source in pairs(snapshot, "inreceptacle")
                     if obj.startswith(target + " "))
    if not sources:
        raise ValueError(f"No recoverable {target} source")
    # Prefer current observable pickup, then an open source, then a closed source.
    available = snapshot["admissible_commands"]
    sources.sort(key=lambda item: (f"take {item[0]} from {item[1]}" not in available,
                                  any(item[1] == row[0] for row in pairs(snapshot, "openable")), item))
    obj, source = sources[0]
    actions = []
    if f"go to {source}" in available:
        actions.append(f"go to {source}")
    if any(source == row[0] for row in pairs(snapshot, "openable")) and not any(source == row[0] for row in pairs(snapshot, "opened")):
        actions.append(f"open {source}")
    actions.append(f"take {obj} from {source}")
    return actions, sources


def search_rubric(cp, snapshot, spec):
    target = spec["object"]
    supporting_types = {rtype for rtype, otype in pairs(snapshot, "cancontain") if otype == target + "type"}
    plausible = sorted(receptacle for receptacle, rtype in pairs(snapshot, "receptacletype") if rtype in supporting_types)
    visited = {row["action"][6:] for row in cp["history"]
               if row["action"].startswith("go to ") and row["observation"] != "Nothing happens."}
    fresh = [receptacle for receptacle in plausible if receptacle not in visited]
    plan, sources = pick_up_plan(snapshot, target)
    strategy = [exact(f"go to {r}") for r in fresh]
    # Opening a fresh candidate can be an independent strategy correction after
    # navigating there; all frozen candidates support the target object type.
    openable = {row[0] for row in pairs(snapshot, "openable")}
    strategy += [exact(f"open {r}") for r in fresh if r in openable]
    recovery = [exact(f"take {obj} from {source}",
                      required_facts=[f"holds(agent1 : agent, {obj}: object)"])
                for obj, source in sources]
    strategy += recovery
    visible_pickups = [rule for rule in recovery if any(re.fullmatch(rule["action_regex"], action)
                       for action in snapshot["admissible_commands"])]
    if visible_pickups:
        # This target is already visible and reachable; more searching alone
        # does not repair the immediate failure to collect it.
        strategy = visible_pickups
        recovery = strategy.copy()
        fresh = []
    directions = ["Execute a valid pickup of the task object already visible and reachable; merely searching elsewhere does not repair this immediate collection stall."] if visible_pickups else [
        "Navigate to a previously unvisited receptacle able to contain the target object, then inspect/open it; or execute a valid pickup of a target object.",
        "New candidate receptacles frozen from official cancontain/receptacletype facts: " + ", ".join(fresh),
        "Arbitrary navigation to already searched or incompatible receptacles does not count."]
    return {
        "evidence": {"failed_action": cp["history"][-1]["action"],
                     "observation": cp["history"][-1]["observation"], "cause": spec["cause"],
                     "state_unchanged_by_failure": cp["history"][-1]["before_state_hash"] == cp["history"][-1]["state_hash"]},
        "blocked_subgoal": spec["blocked"],
        "detection_targets": [target, target.replace("dishsponge", "dish sponge").replace("soapbar", "soap bar").replace("spraybottle", "spray bottle"),
                              cp["history"][-1]["action"], cp["history"][-1]["action"].split(" to ")[-1]],
        "acceptable_directions": directions,
        "strategy_rules": strategy,
        "recovery_rules": recovery,
        "recovery_state": f"A valid pickup has made the agent hold a {target}; merely changing locations or declaring a new plan is insufficient.",
        "probe_actions": plan,
        "rubric_method": "Cause-specific annotation with official symbolic affordances; fixed before A/B; oracle state never added to model prompt.",
    }


def special_rubric(cp):
    if "pick_cool_then_place_in_recep-Apple-None-Microwave-5/" in cp["game_id"]:
        put = "move apple 1 to microwave 1"
        return {"evidence": {"failed_action": "take apple 1 from fridge 1", "observation": "Nothing happens.",
                             "cause": "Apple 1 is already held and cool; cooling does not leave it inside the fridge. Agent is at the closed microwave."},
                "blocked_subgoal": "Place the already-held cool apple in the microwave rather than retaking it from the fridge.",
                "detection_targets": ["apple", "fridge", "take apple 1", "already holding"],
                "acceptable_directions": ["Open microwave 1 while retaining the cool apple, or successfully place that cool apple into microwave 1."],
                "strategy_rules": [exact("open microwave 1", required_facts=["holds(agent1 : agent, apple 1: object)", "iscool(apple 1: object)"]),
                                   exact(put, required_facts=["iscool(apple 1: object)"])],
                "recovery_rules": [exact(put, required_facts=["inreceptacle(apple 1: object, microwave 1: receptacle)", "iscool(apple 1: object)"])],
                "recovery_state": "The cool apple is placed in microwave 1, as verified by official facts and successful placement.",
                "probe_actions": ["open microwave 1", put]}
    put = "move watch 2 to sidetable 1"
    return {"evidence": {"failed_action": cp["history"][-1]["action"], "observation": "Nothing happens.",
                         "cause": "This game accepts exactly 'move watch 2 to sidetable 1'; the attempted placement command is invalid. Watch 2 is held at sidetable 1."},
            "blocked_subgoal": "Execute watch placement using the actual environment command 'move watch 2 to sidetable 1'.",
            "detection_targets": ["watch", "sidetable", "period", "punctuation", "syntax", "put watch 2"],
            "acceptable_directions": ["Execute exactly 'move watch 2 to sidetable 1', retaining the same object and destination."],
            "strategy_rules": [exact(put)],
            "recovery_rules": [exact(put, required_facts=["inreceptacle(watch 2: object, sidetable 1: receptacle)"])],
            "recovery_state": "Watch 2 is successfully placed on sidetable 1; official environment success is separately scored.",
            "probe_actions": [put]}


def build(run_dir):
    root = Path(run_dir)
    if (root / "runs.jsonl").exists() or (root / "checkpoints.jsonl").exists():
        raise RuntimeError("Refuse to change preregistered rubrics after freezing or A/B execution")
    candidates = read_jsonl(root / "candidates.jsonl")
    if len(candidates) != 12:
        raise RuntimeError(f"Need exactly 12 pilot candidates, got {len(candidates)}")
    if sum(cp["kind"] == "error" for cp in candidates) != 8:
        raise RuntimeError("Pilot needs eight error checkpoints")
    rubrics, probes = {}, []
    for cp in candidates:
        if cp["kind"] == "normal":
            action = cp["history"][-1]["action"]
            rubrics[cp["checkpoint_id"]] = {
                "evidence": {"prior_action": action, "observation": cp["history"][-1]["observation"],
                             "cause": "Successful navigation with changed official state; no failed or stalled command before checkpoint."},
                "blocked_subgoal": "None; normal control checkpoint.",
                "detection_targets": [action, action.removeprefix("go to ")],
                "normal_window": "3 actions after checkpoint; claims must concern the successful pre-checkpoint command, not genuinely new post-checkpoint failures.",
                "strategy_rules": [], "recovery_rules": [],
            }
            continue
        env = replay_checkpoint(cp)
        try:
            current = env.snapshot
            target = cp["game_id"].split("/")[1].split("-")[1].lower()
            failed = cp["history"][-1]["action"]
            held_targets = [obj for agent, obj in pairs(current, "holds") if obj.startswith(target + " ")]
            if not held_targets and (failed.startswith("go to ") or failed.startswith("take ")):
                repeated = any(s["action"] == failed and s["observation"] != "Nothing happens." for s in cp["history"][:-1])
                if ". " in failed or ";" in failed:
                    cause = "The action concatenates multiple commands into one unsupported command string; the environment rejects the whole string without changing state."
                elif failed.startswith("go to ") and repeated:
                    cause = f"Navigation to the already occupied {failed[6:]} is rejected without new state or task clues; object collection is stalled."
                elif failed.startswith("take "):
                    cause = "The pickup command does not identify an available numbered target object at the stated source; the environment rejects it without changing state."
                else:
                    cause = f"The command {failed!r} is rejected without changing state, stalling acquisition of {target}."
                spec = {"object": target, "cause": cause,
                        "blocked": f"Advance {target} collection using a valid source or a new plausible search location, then acquire the object."}
                rubric = search_rubric(cp, current, spec)
            elif "pick_cool_then_place_in_recep-Apple-None-Microwave-5/" in cp["game_id"] and held_targets == ["apple 1"]:
                rubric = special_rubric(cp)
            elif "pick_and_place_simple-Watch-None-SideTable-222/" in cp["game_id"] and held_targets == ["watch 2"]:
                rubric = special_rubric(cp)
            else:
                raise RuntimeError(f"New error requires cause-specific annotation: {cp['checkpoint_id']} {cp['game_id']} {failed}")
            before_probe = compact(current)
            steps = []
            for action in rubric["probe_actions"]:
                before = current["state_hash"]
                current = env.step(action)
                step = {**compact(current), "action": action, "before_state_hash": before}
                if step["observation"] == "Nothing happens.":
                    raise RuntimeError(f"Recovery probe action rejected: {cp['checkpoint_id']} {action}")
                steps.append(step)
            strategy_verified = any(rule_matches(rule, step) for rule in rubric["strategy_rules"] for step in steps[:5])
            recovery_verified = any(rule_matches(rule, step) for rule in rubric["recovery_rules"] for step in steps[:10])
            if not strategy_verified or not recovery_verified:
                raise RuntimeError(f"Rubric does not recognize actual probe: {cp['checkpoint_id']}")
            rubric["recoverability_verified"] = True
            rubric["verified_utc"] = datetime.now(timezone.utc).isoformat()
            rubric["recovery_probe_id"] = cp["checkpoint_id"] + "-oracle"
            rubric["oracle_actions_excluded_from_model_history"] = True
            rubrics[cp["checkpoint_id"]] = rubric
            probes.append({"checkpoint_id": cp["checkpoint_id"], "probe_id": rubric["recovery_probe_id"],
                           "purpose": "Independent pre-A/B recoverability validation, excluded from benchmark outcomes",
                           "initial_checkpoint_state": before_probe, "steps": steps,
                           "strategy_verified": strategy_verified, "recovery_verified": recovery_verified,
                           "verified_utc": rubric["verified_utc"]})
        finally:
            env.close()
    write_json(root / "rubrics.json", rubrics)
    (root / "recoverability_probes.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probes))
    print(json.dumps({"rubrics": len(rubrics), "verified_error_probes": len(probes), "all_before_ab": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    build(parser.parse_args().run_dir)
