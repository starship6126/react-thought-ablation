#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_ROOT"
mkdir -p artifacts/setup
record_failure() {
  local failed_command=$1
  python3 - "$failed_command" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path('artifacts/setup/alfworld_setup_failures.jsonl')
with p.open('a') as f:
 f.write(json.dumps({'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'failed_command':sys.argv[1], 'retry_command':'bash scripts/setup_alfworld.sh',
  'required_domains':['pypi.org','files.pythonhosted.org','github.com','release-assets.githubusercontent.com']})+'\n')
PY
}
trap 'record_failure "$BASH_COMMAND"' ERR
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user 'uv==0.13.9'
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
uv pip sync --python .venv/bin/python requirements-alfworld.lock.txt
PYTHONPATH=src .venv/bin/python scripts/setup_alfworld.py
