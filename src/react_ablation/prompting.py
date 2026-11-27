"""Only pre-checkpoint thought bodies differ between paired prompts."""
from __future__ import annotations
import hashlib
import json
import re

SYSTEM = '''You control a household robot in a text game. Complete the task using the observations.
For each turn write exactly two lines:
Thought: one short plan sentence, at most 20 words.
Action: one environment command.
Use exact numbered names from observations. Execute only one action per turn. The Action line must contain the command only, with no final period, quotes, or explanation.
Commands include: go to RECEPTACLE; open RECEPTACLE; close RECEPTACLE; take OBJECT from RECEPTACLE; move OBJECT to RECEPTACLE; clean OBJECT with RECEPTACLE; heat OBJECT with RECEPTACLE; cool OBJECT with RECEPTACLE; use OBJECT; look; inventory; examine OBJECT.
Example:
Observation: You are at countertop 1. On it you see an apple 1.
Thought: I will pick up the apple for the task.
Action: take apple 1 from countertop 1
Observation: You pick up the apple 1 from the countertop 1.
Thought: I will go to the target table.
Action: go to diningtable 1
Thought is a brief public plan, not a detailed reasoning trace.'''


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def render_prompt(initial, history, condition="A", checkpoint_length=0):
    if condition not in ("A", "B"):
        raise ValueError(condition)
    body = "Initial observation and task:\n" + initial["observation"] + "\n\nInteraction history:\n"
    spans = []
    for index, step in enumerate(history):
        body += f"\nTurn {index + 1}\nThought: "
        begin = len(body)
        thought = "[MASKED]" if condition == "B" and index < checkpoint_length else step["thought"]
        body += thought
        spans.append((begin, len(body)))
        body += "\nAction: " + step["action"] + "\nObservation: " + step["observation"] + "\n"
    body += "\nContinue with one Thought line and one Action line.\n"
    prefix = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
    prompt = prefix + body + "<|im_end|>\n<|im_start|>assistant\n"
    return prompt, [(a + len(prefix), b + len(prefix)) for a, b in spans]


def assert_mask_invariance(initial, history):
    a, spans = render_prompt(initial, history, "A", len(history))
    b, _ = render_prompt(initial, history, "B", len(history))
    expected = a
    for begin, end in reversed(spans):
        expected = expected[:begin] + "[MASKED]" + expected[end:]
    assert b == expected, "Mask changed bytes outside historical thought bodies"
    assert len(spans) == len(history)
    return {"passed": True, "thought_count": len(spans), "prompt_a_sha256": digest(a),
            "prompt_b_sha256": digest(b), "thought_body_spans_a": spans,
            "invariant": "Exact byte-equivalent text outside pre-checkpoint thought bodies"}


def parse_response(text):
    match = re.fullmatch(r"\s*Thought:\s*([^\n]+)\nAction:\s*([^\n]+)\s*", text)
    if not match:
        return {"thought": "", "action": "", "parse_ok": False, "raw_response": text}
    return {"thought": match[1].strip(), "action": match[2].strip(), "parse_ok": True,
            "thought_word_count": len(match[1].split()), "raw_response": text}
