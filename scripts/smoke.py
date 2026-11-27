"""Real environment/model integration gate; run before collecting a pilot."""
import json
from pathlib import Path
from statistics import mean
from react_ablation.environment import AlfworldSession, list_games, game_id
from react_ablation.experiment import take_step, make_checkpoint, replay_checkpoint, now
from react_ablation.model import LocalModel
from react_ablation.prompting import assert_mask_invariance
from react_ablation.storage import write_json

root = Path(__file__).resolve().parents[1]
game = list_games("seen")[0]
seed = 20261127
env = AlfworldSession(game, seed)
initial = env.reset()
history = []
model = LocalModel()
model.begin_session()
for i in range(4):
    row = take_step(env, model, initial, history, "A", 0, seed+i)
    history.append(row)
    print(json.dumps({"thought": row["thought"], "action": row["action"], "observation": row["observation"], "model": row["model"]}), flush=True)
    if row["done"]:
        break
env.close()
cp = make_checkpoint(game, "seen", seed, initial, history, "smoke", None)
replayed = replay_checkpoint(cp)
replayed.close()
result = {"created_utc": now(), "actual_environment": True, "game_id": game_id(game),
          "initial": initial, "history": history, "replay_passed": True,
          "mask_validation": assert_mask_invariance(initial, history),
          "parsing_successes": len(history), "parsing_total": len(history),
          "truncated_prompts": sum(s["model"]["truncated"] for s in history),
          "max_prompt_tokens": max(s["model"]["prompt_tokens"] for s in history),
          "mean_latency_seconds": mean(s["model"]["latency_seconds"] for s in history)}
write_json(root/"artifacts/setup/smoke.json", result)
print(json.dumps({k:v for k,v in result.items() if k not in ("history","initial")}),flush=True)
