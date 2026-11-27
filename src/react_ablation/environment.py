"""Official ALFWorld TextWorld backend with stable observation/state snapshots.

The model sees only ``observation``. Admissible commands, entity mappings and
symbolic facts are audit/scoring evidence and are never supplied to the model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("REACT_ALFWORLD_DATA", ROOT / ".runtime/alfworld"))
GAME_ROOT = DATA_ROOT / "json_2.1.1"
TASK_TYPES = {
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
}
SPLITS = {"seen": "valid_seen", "unseen": "valid_unseen", "train": "train"}


def game_id(game_path):
    """Data-root-independent official game ID, including validation split."""
    path = Path(game_path).resolve()
    try:
        return path.relative_to(GAME_ROOT.resolve()).as_posix()
    except ValueError:
        parts = path.parts
        if "json_2.1.1" not in parts:
            raise ValueError(f"Not an ALFWorld game path: {path}")
        return Path(*parts[parts.index("json_2.1.1") + 1:]).as_posix()


def resolve_game(game_path):
    path = Path(game_path)
    if path.is_file():
        return path.resolve()
    # Recorded absolute paths remain replayable after cloning elsewhere.
    identity = game_id(path) if path.is_absolute() else path.as_posix()
    candidate = GAME_ROOT / identity
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing game {candidate}; run scripts/setup_alfworld.sh")
    return candidate.resolve()


def list_games(split):
    """Use official AlfredTWEnv eligibility: six tasks, non-sliced, solvable."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    base = GAME_ROOT / SPLITS[split]
    if not base.is_dir():
        raise FileNotFoundError(f"Missing {base}; run scripts/setup_alfworld.sh")
    games = []
    for trajectory in sorted(base.rglob("traj_data.json")):
        folder = trajectory.parent
        if "movable" in str(folder) or "Sliced" in str(folder):
            continue
        data = json.loads(trajectory.read_text())
        if data.get("task_type") not in TASK_TYPES:
            continue
        game = folder / "game.tw-pddl"
        if not game.exists():
            continue
        game_data = json.loads(game.read_text())
        if game_data.get("solvable") is True:
            games.append(str(game.resolve()))
    return games


class AlfworldSession:
    """Independent PDDL environment using ALFWorld's official evaluation wrappers."""

    def __init__(self, game_path, seed):
        self.game_path = resolve_game(game_path)
        self.seed = int(seed)
        self.env = None
        self.snapshot = None

    def reset(self):
        self.close()
        random.seed(self.seed)
        np.random.seed(self.seed % 2**32)
        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                                   moves=True, extras=["gamefile"])
        self.env = textworld.start(str(self.game_path), request_infos=infos,
                                  wrappers=[lambda env: AlfredDemangler(env, shuffle=False), AlfredInfos])
        # PDDL transitions and grammar are deterministic for the fixed game;
        # seed() is the official backend API (a no-op in this TextWorld version).
        self.env.seed(self.seed)
        state = self.env.reset()
        self.snapshot = self._snapshot(state, done=bool(state.get("won", False)))
        return self.snapshot

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Call reset before step")
        state, reward, done = self.env.step(action)
        self.snapshot = self._snapshot(state, done=done, reward=reward)
        return self.snapshot

    def _snapshot(self, state, done=False, reward=0):
        records = sorted(
            ({"predicate": p.name,
              "arguments": [{"name": v.name, "type": v.type} for v in p.arguments]}
             for p in state.get("facts", [])),
            key=lambda value: json.dumps(value, sort_keys=True),
        )
        raw_records = sorted(
            ({"predicate": p.name,
              "arguments": [{"name": v.name, "type": v.type} for v in p.arguments]}
             for p in state.get("_facts", [])),
            key=lambda value: json.dumps(value, sort_keys=True),
        )
        canonical = json.dumps(raw_records, sort_keys=True, separators=(",", ":"))
        facts = sorted(str(p) for p in state.get("facts", []))
        entities = {identifier: {"name": info.name, "type": info.type}
                    for identifier, info in sorted(state.get("_entity_infos", {}).items())}
        return {"observation": state.feedback, "state_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                "facts": facts, "fact_records": records,
                "admissible_commands": sorted(state.get("admissible_commands", [])),
                "entities": entities, "won": bool(state.get("won", False)), "done": bool(done),
                "reward": float(reward), "moves": int(state.get("moves", 0))}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
