#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
mkdir -p .runtime/slot-cache
python3 scripts/write_runtime_manifest.py
python3 - <<'PY'
import json,subprocess,time,urllib.request
from pathlib import Path
root=Path.cwd()
manifest=json.loads((root/'artifacts/setup/runtime_manifest.json').read_text())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open('http://127.0.0.1:18080/health',timeout=2) as response:
        pid_file=root/'artifacts/setup/llama-server.pid'
        if not pid_file.exists():
            raise SystemExit('Port 18080 already serves HTTP without our PID file; refusing reuse.')
        pid=int(pid_file.read_text())
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if command != manifest['server_argv']:
            raise SystemExit('Existing server arguments differ from fixed manifest; refusing reuse.')
        print(f'Existing fixed server verified: PID {pid}, CPU only, http://127.0.0.1:18080')
        raise SystemExit(0)
except OSError:
    pass
log=(root/'artifacts/setup/llama-server.log').open('ab')
process=subprocess.Popen(manifest['server_argv'],stdin=subprocess.DEVNULL,stdout=log,
                         stderr=subprocess.STDOUT,start_new_session=True)
(root/'artifacts/setup/llama-server.pid').write_text(str(process.pid)+'\n')
for _ in range(60):
    if process.poll() is not None:
        raise SystemExit('llama-server failed; see artifacts/setup/llama-server.log')
    try:
        with opener.open('http://127.0.0.1:18080/health',timeout=1) as response:
            print(f'Ready: PID {process.pid}, CPU only, http://127.0.0.1:18080')
            break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit('Server did not become healthy in 30 seconds; inspect logs.')
PY
