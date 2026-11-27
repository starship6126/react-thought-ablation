"""Predeclared behavior scoring and condition-blinded provisional language review.

No language model is used by this module. The language heuristics are deliberately
conservative and remain provisional until the editable blind-review fields are filled.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

from .storage import read_jsonl, write_json

METRICS = ("detection_3", "strategy_5", "recovery_10", "success_20", "false_positive_3")
LANGUAGE = {"detection_3", "false_positive_3"}
FAILURE_OBSERVATION = re.compile(r"nothing happens|can't|cannot|don't understand|not a valid|invalid command|you have to", re.I)
ERROR_LANGUAGE = re.compile(
    r"\b(?:failed|failure|fails?|cannot|can't|couldn't|unable|blocked|stuck|wrong|invalid|"
    r"unsuccessful|unavailable|missing|not found|not there|not here|not open|not holding|"
    r"need to open|must open|must first|need to first|doesn't work|didn't work)\b"
    r"|nothing happens|nothing happened|\balready (?:at|holding|carrying|open)\b", re.I)


def _normal(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _result(score=0, evidence="No qualifying evidence in the registered window.", provisional=False):
    return {"score": int(bool(score)), "evidence": evidence, "provisional": provisional}


def _specific_reference(text, checkpoint):
    targets = checkpoint.get("rubric", {}).get("detection_targets", [])
    normalized = _normal(text)
    return any(_normal(target) and _normal(target) in normalized for target in targets)


def _language_detection(checkpoint, steps):
    for index, step in enumerate(steps[:3], 1):
        thought = step.get("thought", "")
        if ERROR_LANGUAGE.search(thought) and _specific_reference(thought, checkpoint):
            return _result(1, f"Turn {index} specifically acknowledges a blocked subgoal: {thought}", True)
    return _result(0, "No specific failure/subgoal acknowledgment within 3 actions; generic retry language does not count.", True)


def _facts(step):
    return {json.dumps(fact, sort_keys=True) if not isinstance(fact, str) else fact for fact in step.get("facts", [])}


def rule_matches(rule, step):
    """Match preregistered actions and, where required, actual state/fact changes."""
    action = step.get("action", "")
    if rule.get("action_regex") and re.fullmatch(rule["action_regex"], action, re.I) is None:
        return False
    if rule.get("require_state_change", False):
        before, after = step.get("before_state_hash"), step.get("state_hash")
        if not before or not after or before == after:
            return False
    # A matching rejected command cannot count as executed correction or recovery.
    if FAILURE_OBSERVATION.search(step.get("observation", "")):
        return False
    facts = _facts(step)
    if any(fact not in facts for fact in rule.get("required_facts", [])):
        return False
    if any(fact in facts for fact in rule.get("forbidden_facts", [])):
        return False
    if rule.get("observation_regex") and not re.search(rule["observation_regex"], step.get("observation", ""), re.I):
        return False
    return bool(rule.get("action_regex") or rule.get("required_facts") or rule.get("observation_regex"))


def _behavior(checkpoint, steps, key, window):
    rules = checkpoint.get("rubric", {}).get(key, [])
    for index, step in enumerate(steps[:window], 1):
        for rule_index, rule in enumerate(rules, 1):
            if rule_matches(rule, step):
                return _result(1, f"Turn {index}: executed {step['action']!r}; frozen {key} rule {rule_index} matched; observation={step.get('observation', '')}")
    return _result()


def _normal_false_positive(checkpoint, steps):
    # At turn 1 all failure claims concern pre-checkpoint evidence. At later turns
    # the model may correctly identify a NEW error: exclude those from this metric.
    last_prior = checkpoint.get("history", [{}])[-1]
    prior_action = _normal(last_prior.get("action", ""))
    for index, step in enumerate(steps[:3]):
        thought = step.get("thought", "")
        if not ERROR_LANGUAGE.search(thought):
            continue
        earlier = steps[:index]
        if any(FAILURE_OBSERVATION.search(row.get("observation", "")) for row in earlier):
            continue
        # Require a concrete past command or supplied normal-specific target,
        # not a prospective prerequisite such as "open the fridge before ...".
        retrospective = re.search(r"failed|failure|unsuccessful|wrong|invalid|didn't|couldn't|nothing happened|stuck|blocked", thought, re.I)
        specific = (prior_action and prior_action in _normal(thought)) or _specific_reference(thought, checkpoint)
        if retrospective and specific:
            return _result(1, f"Turn {index+1} alleges a specific pre-checkpoint failure without observed failure: {thought}", True)
    return _result(0, "No specific false pre-checkpoint failure claim detected; new post-checkpoint failures are excluded.", True)


def score_checkpoint(checkpoint, run):
    steps = (run or {}).get("steps", [])[:20]
    result = {metric: _result(provisional=metric in LANGUAGE) for metric in METRICS}
    if checkpoint["kind"] == "error":
        result["detection_3"] = _language_detection(checkpoint, steps)
        result["strategy_5"] = _behavior(checkpoint, steps, "strategy_rules", 5)
        result["recovery_10"] = _behavior(checkpoint, steps, "recovery_rules", 10)
    else:
        result["false_positive_3"] = _normal_false_positive(checkpoint, steps)
    won_turns = [i for i, s in enumerate(steps, 1) if s.get("won")]
    result["success_20"] = _result(bool(won_turns), f"Official environment won=True at turn {won_turns[0]}." if won_turns else "No official won=True within 20 actions.")
    return result


def paired_bootstrap(checkpoints, differences, samples=10000, seed=20261127):
    """Resample games, retaining every checkpoint and both conditions in each game."""
    groups = {}
    for checkpoint, difference in zip(checkpoints, differences):
        groups.setdefault(checkpoint["game_id"], []).append(float(difference))
    if not groups:
        return (None, None)
    values = list(groups.values())
    sums = np.array([sum(group) for group in values])
    counts = np.array([len(group) for group in values])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(samples, len(values)))
    rates = sums[choices].sum(axis=1) / counts[choices].sum(axis=1) * 100
    return tuple(float(x) for x in np.quantile(rates, [0.025, 0.975]))


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path, rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, output.getvalue())


def _load_review(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["review_id"]: row for row in csv.DictReader(handle)}


def _blind_text(value):
    # Prefix history never enters the review transcript. Remove any direct mask
    # literal copied by the model, while the complete raw text remains in runs.
    return str(value).replace("[MASKED]", "[history reference]")


def score_directory(run_dir, bootstrap_samples=10000, seed=20261127):
    root = Path(run_dir)
    checkpoints = read_jsonl(root / "checkpoints.jsonl")
    raw_runs = read_jsonl(root / "runs.jsonl")
    runs = {}
    for run in raw_runs:
        key = (run["checkpoint_id"], run["condition"])
        if key in runs:
            raise ValueError(f"Duplicate condition run: {key}")
        runs[key] = run
    old_reviews = _load_review(root / "blind_review.csv")
    scores, review_rows, mapping = {}, [], []
    for cp in checkpoints:
        for condition in ("A", "B"):
            key = (cp["checkpoint_id"], condition)
            run = runs.get(key)
            score = score_checkpoint(cp, run)
            review_id = hashlib.sha256(f"review-v1|{root.name}|{key[0]}|{condition}".encode()).hexdigest()[:20]
            mapping.append({"review_id": review_id, "checkpoint_id": key[0], "condition": condition})
            previous = old_reviews.get(review_id, {})
            reviewed_metric = "detection_3" if cp["kind"] == "error" else "false_positive_3"
            override = previous.get("reviewer_" + reviewed_metric, "").strip()
            if override:
                if override not in ("0", "1") or not previous.get("reviewer_reason", "").strip():
                    raise ValueError("Review overrides need a 0/1 score and reviewer_reason")
                if run is None:
                    raise ValueError("Cannot review an unexecuted run")
                score[reviewed_metric] = _result(int(override), previous["reviewer_reason"], False)
            scores[key] = score
            prior = cp.get("history", [{}])[-1]
            transcript = [{"turn": i, "thought": _blind_text(step.get("thought", "")),
                           "action": step.get("action", ""), "observation": step.get("observation", "")}
                          for i, step in enumerate((run or {}).get("steps", [])[:3], 1)]
            row = {"review_id": review_id, "checkpoint_kind": cp["kind"],
                   "run_complete": bool(run and run.get("status") == "completed"),
                   "failure_evidence": json.dumps(cp.get("rubric", {}).get("evidence", cp.get("failure")), ensure_ascii=False),
                   "blocked_subgoal": cp.get("rubric", {}).get("blocked_subgoal", ""),
                   "prior_action": prior.get("action", ""), "prior_observation": prior.get("observation", ""),
                   "turns_1_3": json.dumps(transcript, ensure_ascii=False),
                   "provisional_detection_3": _language_detection(cp, (run or {}).get("steps", []))["score"] if cp["kind"] == "error" else "",
                   "provisional_false_positive_3": _normal_false_positive(cp, (run or {}).get("steps", []))["score"] if cp["kind"] == "normal" else "",
                   "provisional_reason": _blind_text(score_checkpoint(cp, run)[reviewed_metric]["evidence"]),
                   "reviewer_detection_3": previous.get("reviewer_detection_3", ""),
                   "reviewer_false_positive_3": previous.get("reviewer_false_positive_3", ""),
                   "reviewer_reason": previous.get("reviewer_reason", ""),
                   "reviewer_id": previous.get("reviewer_id", "")}
            review_rows.append(row)
    review_rows.sort(key=lambda row: row["review_id"])
    review_fields = ["review_id", "checkpoint_kind", "run_complete", "failure_evidence", "blocked_subgoal", "prior_action", "prior_observation", "turns_1_3", "provisional_detection_3", "provisional_false_positive_3", "provisional_reason", "reviewer_detection_3", "reviewer_false_positive_3", "reviewer_reason", "reviewer_id"]
    _write_csv(root / "blind_review.csv", review_rows, review_fields)
    write_json(root / "review_mapping.json", mapping)
    completed_pairs = sum(all(runs.get((cp["checkpoint_id"], c), {}).get("status") == "completed" for c in ("A", "B")) for cp in checkpoints)
    report = []
    report_metrics = [(metric, metric, None if metric == "success_20" else "normal" if metric == "false_positive_3" else "error") for metric in METRICS]
    report_metrics += [("success_20_error", "success_20", "error"), ("success_20_normal", "success_20", "normal")]
    for metric_index, (metric, source_metric, kind) in enumerate(report_metrics):
        selected = [cp for cp in checkpoints if kind is None or cp["kind"] == kind]
        a = [scores[(cp["checkpoint_id"], "A")][source_metric]["score"] for cp in selected]
        b = [scores[(cp["checkpoint_id"], "B")][source_metric]["score"] for cp in selected]
        n = len(selected)
        low, high = paired_bootstrap(selected, [x-y for x, y in zip(a, b)], bootstrap_samples, seed + metric_index)
        provisional = any(scores[(cp["checkpoint_id"], c)][source_metric]["provisional"] for cp in selected for c in ("A", "B"))
        report.append({"metric": metric, "a_numerator": sum(a), "a_denominator": n,
                       "b_numerator": sum(b), "b_denominator": n,
                       "a_percent": 100 * sum(a) / n if n else None,
                       "b_percent": 100 * sum(b) / n if n else None,
                       "a_minus_b_pp": 100 * (sum(a)-sum(b)) / n if n else None,
                       "ci95_low_pp": low, "ci95_high_pp": high,
                       "a_only": sum(x and not y for x, y in zip(a, b)),
                       "b_only": sum(y and not x for x, y in zip(a, b)),
                       "paired_games": len({cp["game_id"] for cp in selected}),
                       "completed_pairs": completed_pairs, "assigned_pairs": len(checkpoints),
                       "score_status": "provisional_language" if provisional else "reviewed_language" if metric in LANGUAGE else "automatic_environment_behavior"})
    fields = list(report[0])
    _write_csv(root / "report.csv", report, fields)
    write_json(root / "scores.json", [{"checkpoint_id": key[0], "condition": key[1], "scores": score} for key, score in scores.items()])
    _write_report(root, report, checkpoints, raw_runs, completed_pairs, bootstrap_samples, seed)
    return report


def _write_report(root, rows, checkpoints, runs, completed_pairs, bootstrap_samples, seed):
    def fmt(value):
        return "NA" if value is None else f"{value:.2f}"
    text = [f"# ReAct thought-history ablation: {root.name}", "",
            f"Actual ALFWorld paired runs completed: **{completed_pairs}/{len(checkpoints)}**. "
            f"Assigned error checkpoints: {sum(cp['kind']=='error' for cp in checkpoints)}; normal checkpoints: {sum(cp['kind']=='normal' for cp in checkpoints)}.", "",
            "A retains all pre-checkpoint thought bodies. B replaces only those bodies with `[MASKED]`. "
            "Post-checkpoint thoughts remain in each condition's own history. Each independent environment session replays the original game and seed; at most 20 actions are generated by the fixed local CPU model.", "",
            "| Metric | A numerator/denominator (%) | B numerator/denominator (%) | A−B pp | Paired game bootstrap 95% CI (pp) | A only | B only | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        text.append(f"| {row['metric']} | {row['a_numerator']}/{row['a_denominator']} ({fmt(row['a_percent'])}) | {row['b_numerator']}/{row['b_denominator']} ({fmt(row['b_percent'])}) | {fmt(row['a_minus_b_pp'])} | [{fmt(row['ci95_low_pp'])}, {fmt(row['ci95_high_pp'])}] | {row['a_only']} | {row['b_only']} | {row['score_status']} |")
    depths = {}
    for kind in ("error", "normal"):
        values = [len(cp.get("history", [])) for cp in checkpoints if cp["kind"] == kind]
        if values:
            depths[kind] = f"{kind}: min {min(values)}, median {np.median(values):g}, max {max(values)}"
    text += ["", "Pre-checkpoint history depth in thought/action turns — " + "; ".join(depths.values()) + ". "
             "These depths delimit how much prior thought information this pilot actually removes.", "",
             "`[MASKED]` replaces variable-length thought bodies with a fixed marker. All other prompt text is invariant, "
             "but token counts and subsequent token positions can differ. This is an effect of the specified masking intervention; "
             "there is no length-matched control."]
    preflight_path = root / "preflight_old_placement_syntax" / "status.json"
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text())
        discarded = read_jsonl(preflight_path.parent / "candidates.jsonl")
        text += ["", f"Preflight diagnostics retained in `preflight_old_placement_syntax/` contain {len(discarded)} discarded candidates "
                 f"and {preflight.get('benchmark_pairs_executed', 'unknown')} benchmark A/B pairs. "
                 "A recovery probe found an unsupported placement command in the shared prompt; it was corrected to the official `move OBJECT to RECEPTACLE` grammar. "
                 "All reported checkpoints were recollected with that corrected prompt before freezing and A/B execution. The model and decoding settings were unchanged."]
    text += ["", "Detection requires a specific failed command or blocked subgoal within 3 actions. Generic retry language scores 0. "
             "Strategy correction is an actually executed action matching a frozen cause-specific rule within 5 actions, independent of verbal detection. "
             "Recovery uses frozen environment/action predicates within 10 actions. Task success uses the environment's official `won` flag within 20 actions. "
             "`success_20` includes all assigned checkpoints; `success_20_error` and `success_20_normal` report the two checkpoint strata separately. "
             "Normal-case false positives concern pre-checkpoint history within 3 actions; newly observed post-checkpoint failures do not count.", "",
             "Language scores are conservative lexical provisional scores unless an explicit blinded reviewer override is entered in `blind_review.csv`. "
             "That CSV contains no condition name or pre-checkpoint thought body. `review_mapping.json` must be withheld from reviewers. "
             "The heuristic can miss paraphrases and cannot replace human adjudication; its evidence is retained for inspection.", "",
             f"The percentile bootstrap resamples whole games with replacement ({bootstrap_samples:,} draws, base seed {seed}; metric seeds increment by table row), retaining paired conditions and all checkpoints in each sampled game. "
             "The error-metric denominator is every assigned error checkpoint in each condition. Missing runs are zero, never dropped; incomplete reporting is not a completed benchmark. "
             "A only/B only are paired discordance counts, also reported for correction and detection, not only task success.", ""]
    steps = [step for run in runs for step in run.get("steps", [])]
    if steps:
        latencies = [step["model"]["latency_seconds"] for step in steps if "model" in step]
        tokens = [step["model"]["prompt_tokens"] for step in steps if "model" in step]
        parsed = sum(bool(step.get("parse_ok")) for step in steps)
        text += [f"Executed continuation actions: {len(steps)}. Parsed responses: {parsed}/{len(steps)}. "
                 f"Reported prompt truncations: {sum(bool(step.get('model', {}).get('truncated')) for step in steps)}."]
        if latencies:
            text.append(f"Response latency: median {np.median(latencies):.2f}s, p95 {np.quantile(latencies, .95):.2f}s; prompt tokens maximum {max(tokens)}.")
    for name in ("runtime_manifest.json", "model_manifest.json"):
        path = root / name
        if path.exists():
            text += ["", f"Pinned runtime metadata (`{name}`):", "```json", path.read_text().strip(), "```"]
            break
    text += ["", "This is an exploratory pilot of Qwen2.5-1.5B-Instruct GGUF Q4_K_M on the official ALFWorld text environment. "
             "A small, selected set of recoverable checkpoints and normal controls does not establish effects for other models, environments, or the full unseen benchmark. "
             "Wide or degenerate bootstrap intervals with few games should not be interpreted as definitive evidence of equivalence or causality outside this intervention. "
             "The unseen 134-game main experiment is a separate run targeting 80 error and 20 normal checkpoints; these pilot values must not be substituted for it.", ""]
    _atomic_text(root / "report.md", "\n".join(text))
