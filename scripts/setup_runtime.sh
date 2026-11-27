#!/usr/bin/env bash
# Public, pinned downloads only. No model API key is used.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
mkdir -p .runtime artifacts/setup
RUNTIME_RETRY='bash scripts/setup_runtime.sh'
trap 'code=$?; printf "exit=%s\ncommand=%s\nretry=%s\ndomains=github.com,codeload.github.com,huggingface.co,cdn-lfs.huggingface.co,cas-bridge.xethub.hf.co,transfer.xethub.hf.co,pypi.org,files.pythonhosted.org\n" "$code" "$BASH_COMMAND" "$RUNTIME_RETRY" >> artifacts/setup/runtime_download_failures.log; exit "$code"' ERR

python3 - <<'PY'
import datetime,json,os,platform,shutil,subprocess
from pathlib import Path
p=Path('artifacts/setup/resources_current.json')
def output(command):
    return subprocess.run(command,text=True,capture_output=True,check=True).stdout
def optional(path):
    p=Path(path)
    return p.read_text().strip() if p.exists() else None
data={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'platform':platform.platform(),'cpu_count':os.cpu_count(),
      'cpu_affinity':len(os.sched_getaffinity(0)),'cpu':output(['lscpu']),
      'memory':output(['free','-b']),'disk':dict(zip(('total','used','free'),shutil.disk_usage('.'))),
      'cgroup_cpu_max':optional('/sys/fs/cgroup/cpu.max'),
      'cgroup_memory_max':optional('/sys/fs/cgroup/memory.max')}
p.write_text(json.dumps(data,indent=2)+'\n')
initial=Path('artifacts/setup/resources.json')
if not initial.exists(): initial.write_text(p.read_text())
PY

HF_REVISION=91cad51170dc346986eccefdc2dd33a9da36ead9
WEIGHT_SHA256=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
LLAMA_REVISION=5d7bb89f4719243dc52fe387bcb67c3518256565
WEIGHT=.runtime/qwen2.5-1.5b-instruct-q4_k_m.gguf
if [ ! -f "$WEIGHT" ]; then
  curl -fL --retry 3 --continue-at - "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$HF_REVISION/qwen2.5-1.5b-instruct-q4_k_m.gguf" -o "$WEIGHT.partial"
  printf '%s  %s\n' "$WEIGHT_SHA256" "$WEIGHT.partial" | sha256sum --check
  mv "$WEIGHT.partial" "$WEIGHT"
fi
printf '%s  %s\n' "$WEIGHT_SHA256" "$WEIGHT" | sha256sum --check
if [ ! -d .runtime/llama.cpp/.git ]; then
  git clone https://github.com/ggml-org/llama.cpp.git .runtime/llama.cpp
fi
git -C .runtime/llama.cpp checkout --detach "$LLAMA_REVISION"
if ! command -v cmake >/dev/null; then
  python3 -m venv .runtime/build-tools
  .runtime/build-tools/bin/pip install cmake==4.4.5
  export PATH="$PROJECT_ROOT/.runtime/build-tools/bin:$PATH"
fi
cmake -S .runtime/llama.cpp -B .runtime/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF > artifacts/setup/cmake-configure.log 2>&1
cmake --build .runtime/llama.cpp/build --target llama-server -j 4 > artifacts/setup/cmake-build.log 2>&1
python3 scripts/write_runtime_manifest.py
printf 'Runtime ready. Start with: bash scripts/start_server.sh\n'
