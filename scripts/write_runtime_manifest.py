"""Record a stable, keyless local model identity after setup."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = ROOT / '.runtime/qwen2.5-1.5b-instruct-q4_k_m.gguf'
BINARY = ROOT / '.runtime/llama.cpp/build/bin/llama-server'
EXPECTED_SHA = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
EXPECTED_COMMIT = '5d7bb89f4719243dc52fe387bcb67c3518256565'


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def output(args):
    p = subprocess.run(args, text=True, capture_output=True, check=True)
    return (p.stdout + p.stderr).strip()


weight_hash = file_hash(WEIGHT)
commit = output(['git', '-C', str(ROOT / '.runtime/llama.cpp'), 'rev-parse', 'HEAD'])
assert weight_hash == EXPECTED_SHA, 'Unexpected model weights'
assert commit == EXPECTED_COMMIT, 'Unexpected llama.cpp revision'
args = [str(BINARY), '--model', str(WEIGHT), '--host', '127.0.0.1', '--port', '18080',
        '--ctx-size', '16384', '--parallel', '1', '--threads', '4', '--threads-batch', '4',
        '--n-gpu-layers', '0', '--no-context-shift', '--no-cache-prompt',
        '--no-cache-idle-slots', '--offline', '--slot-save-path', str(ROOT / '.runtime/slot-cache')]
manifest = {
    'identity': {
        'model_repo': 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
        'model_revision': '91cad51170dc346986eccefdc2dd33a9da36ead9',
        'weights_sha256': weight_hash,
        'llama_commit': commit,
        'server_config': {
            'host': '127.0.0.1', 'port': 18080, 'cpu_threads': 4,
            'context_size': 16384, 'parallel': 1, 'gpu_layers': 0,
            'context_shift': False, 'prompt_cache': False, 'offline': True,
            'request_prompt_cache': True, 'session_cache_erased': True,
            'slot_actions_enabled': True,
        },
    },
    'model': 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    'hf_revision': '91cad51170dc346986eccefdc2dd33a9da36ead9',
    'quantization': 'Q4_K_M', 'weights_path': str(WEIGHT),
    'weights_filename': WEIGHT.name, 'weights_sha256': weight_hash,
    'weights_bytes': WEIGHT.stat().st_size, 'llama_cpp_commit': commit,
    'llama_cpp_version': output([str(BINARY), '--version']),
    'server_binary_path': str(BINARY), 'server_binary_sha256': file_hash(BINARY),
    'server_argv': args, 'host': '127.0.0.1', 'port': 18080,
    'cpu_threads': 4, 'context_size': 16384, 'parallel': 1,
    'gpu_layers': 0, 'context_shift': False, 'prompt_cache': False,
    'request_prompt_cache': True, 'session_cache_erased': True,
    'cache_numerics_caveat': 'llama.cpp notes that KV reuse can change batch sizes and floating-point results; identical policy is fixed for A and B, with an explicit slot erase before each independent session.',
    'external_model_api_calls': False, 'api_key_required': False,
    'build': {'type': 'Release', 'GGML_CUDA': False, 'GGML_NATIVE': True,
              'GGML_BLAS': False, 'GGML_OPENMP': True},
    'public_download_domains': ['github.com', 'huggingface.co', 'cas-bridge.xethub.hf.co',
                                'transfer.xethub.hf.co', 'pypi.org', 'files.pythonhosted.org'],
}
path = ROOT / 'artifacts/setup/runtime_manifest.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
(ROOT / 'artifacts/setup/server_argv.json').write_text(json.dumps(args, indent=2) + '\n')
print(path)
