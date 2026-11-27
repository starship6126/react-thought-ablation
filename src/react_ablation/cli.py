import argparse
import json
from pathlib import Path
from .experiment import collect, freeze, run_pairs


def main():
    parser = argparse.ArgumentParser(description="Actual local-model ALFWorld thought ablation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=["seen", "unseen"], default="seen")
    p.add_argument("--errors", type=int, default=8)
    p.add_argument("--normals", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261127)
    p.add_argument("--all-games", action="store_true")
    p = sub.add_parser("freeze")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--rubrics", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=20261127)
    p = sub.add_parser("score")
    p.add_argument("--run-dir", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "collect":
        rows = collect(**args)
        print(json.dumps({"candidates": len(rows)}))
    elif command == "freeze":
        rows = freeze(args["run_dir"], args["rubrics"])
        print(json.dumps({"frozen": len(rows)}))
    elif command == "run":
        print(json.dumps({"completed_pairs": run_pairs(**args)}))
    else:
        from .scoring import score_directory
        score_directory(Path(args["run_dir"]))


if __name__ == "__main__":
    main()
