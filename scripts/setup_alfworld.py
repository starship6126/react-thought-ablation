"""Download only official ALFWorld text data, verify archives, and audit replay."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".runtime/alfworld"
ARTIFACTS = ROOT / "artifacts/setup"
SOURCES = [
    ("https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
     "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc"),
    ("https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
     "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"),
]


def save(name, data):
    (ARTIFACTS / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    archives = []
    for url, expected in SOURCES:
        archive = DATA / url.rsplit("/", 1)[-1]
        if not archive.exists():
            cmd = ["curl", "--fail", "--location", "--retry", "3", "--output", str(archive) + ".partial", url]
            try:
                subprocess.run(cmd, check=True)
            except Exception as exc:
                with (ARTIFACTS / "alfworld_download_failures.jsonl").open("a") as f:
                    f.write(json.dumps({"url": url, "failed_command": cmd, "error": str(exc),
                                        "domains": ["github.com", "release-assets.githubusercontent.com"],
                                        "retry_command": "bash scripts/setup_alfworld.sh"}) + "\n")
                raise
            archive.with_suffix(archive.suffix + ".partial").replace(archive)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Archive hash mismatch: {archive}")
        # Official archives are verified above; still reject escaping member names.
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (DATA / member.filename).resolve()
                if DATA.resolve() not in target.parents and target != DATA.resolve():
                    raise RuntimeError("Unsafe archive path")
                if not target.exists():
                    z.extract(member, DATA)
        archives.append({"url": url, "sha256": actual, "bytes": archive.stat().st_size})

    from react_ablation.environment import AlfworldSession, list_games, game_id
    counts = {s: len(list_games(s)) for s in ["seen", "unseen"]}
    assert counts == {"seen": 140, "unseen": 134}, counts
    game = list_games("seen")[0]
    seed = 20261127
    actions = ["go to bed 1", "look", "take alarmclock 1 from bed 1", "inventory", "go to desk 1"]
    with AlfworldSession(game, seed) as env:
        trajectory = [env.reset()]
        for action in actions:
            trajectory.append(env.step(action))
    with AlfworldSession(game, seed) as env:
        replayed = [env.reset()]
        for action in actions:
            replayed.append(env.step(action))
    checked = ["observation", "state_hash", "facts", "admissible_commands", "won", "done"]
    for actual, expected in zip(replayed, trajectory):
        for field in checked:
            assert actual[field] == expected[field], field
    save("environment_probe.json", {"actual_environment": True, "model_used": False,
         "purpose": "Official backend and deterministic replay validation; excluded from benchmark",
         "game_id": game_id(game), "game_sha256": hashlib.sha256(Path(game).read_bytes()).hexdigest(),
         "seed": seed, "actions": actions, "replay_passed": True, "fields_checked": checked,
         "trajectory": [{k: s[k] for k in checked} for s in trajectory]})
    versions = {name: metadata.version(name) for name in ["alfworld", "textworld", "fast-downward-textworld", "numpy", "requests"]}
    save("environment_manifest.json", {"recorded_utc": datetime.now(timezone.utc).isoformat(),
         "python": platform.python_version(), "packages": versions, "archives": archives,
         "split_counts": counts, "wrapper": "Official AlfredDemangler(shuffle=False), AlfredInfos",
         "backend": "textworld.envs.pddl.PddlEnv via textworld.start",
         "seed_semantics": "Fixed game + deterministic PDDL; Python and NumPy seeded; backend seed() is a no-op",
         "success": "Official TextWorld state.won from PDDL check_goal()",
         "model_observation_fields": ["observation"],
         "audit_only_fields": ["facts", "fact_records", "entities", "admissible_commands", "state_hash"]})
    print(json.dumps({"split_counts": counts, "replay_passed": True, "packages": versions}))


if __name__ == "__main__":
    main()
