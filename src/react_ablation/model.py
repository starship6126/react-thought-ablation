"""Keyless HTTP client restricted to a loopback llama.cpp server."""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import requests
from .prompting import digest, parse_response

GRAMMAR = r'''root ::= "Thought: " sentence "\nAction: " command "\n"
sentence ::= [^\n\r]{1,160}
command ::= [^\n\r]{1,120}'''

@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://127.0.0.1:18080"
    context_size: int = 16384
    n_predict: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    cache_prompt: bool = True


class LocalModel:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        url = urlparse(self.config.base_url)
        if url.hostname != "127.0.0.1" or url.scheme != "http":
            raise ValueError("Only the authorized loopback CPU server is supported")
        self.http = requests.Session()
        # Loopback must not be sent to any HTTP proxy; no authorization header is used.
        self.http.trust_env = False
        self.first_request = True

    def post(self, path, data):
        result = self.http.post(self.config.base_url + path, json=data, timeout=300)
        result.raise_for_status()
        return result.json()

    def begin_session(self):
        """Erase every prior session's KV state; reuse only this session's prefix."""
        result = self.post("/slots/0?action=erase", {})
        if "n_erased" not in result:
            raise RuntimeError("Server did not confirm independent session cache erasure")
        self.first_request = True
        return result

    def verify_identity(self, identity):
        result = self.http.get(self.config.base_url + "/props", timeout=20)
        result.raise_for_status()
        props = result.json()
        from pathlib import Path
        assert Path(props["model_path"]).name == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        assert props["total_slots"] == 1
        assert props["default_generation_settings"]["n_ctx"] == self.config.context_size
        assert identity["llama_commit"][:7] in props["build_info"]
        assert "Q4_K" in props["model_ftype"]
        assert not props["default_generation_settings"]["params"].get("lora")
        return {key: props[key] for key in ("model_path", "model_ftype", "build_info", "total_slots")}

    def generate(self, prompt, seed):
        cfg = asdict(self.config)
        token_count = len(self.post("/tokenize", {"content": prompt, "add_special": True})["tokens"])
        if token_count + cfg["n_predict"] + 32 >= cfg["context_size"]:
            raise RuntimeError(f"Context budget exceeded: {token_count}; no truncation allowed")
        payload = {k: v for k, v in cfg.items() if k not in ("base_url", "context_size")}
        payload.update(prompt=prompt, seed=seed, stream=False,
                       grammar=GRAMMAR, stop=["<|im_end|>"], id_slot=0)
        start = time.monotonic()
        response = self.post("/completion", payload)
        elapsed = time.monotonic() - start
        if response.get("truncated", False):
            raise RuntimeError("llama.cpp truncated a prompt")
        cache_tokens = response.get("timings", {}).get("cache_n", 0)
        if self.first_request and cache_tokens != 0:
            raise RuntimeError("First request reused a prior session's KV cache")
        self.first_request = False
        parsed = parse_response(response["content"])
        parsed["model"] = {"latency_seconds": elapsed, "prompt_tokens": token_count,
                           "prompt_sha256": digest(prompt), "seed": seed,
                           "truncated": response.get("truncated", False),
                           "tokens_predicted": response.get("tokens_predicted"),
                           "tokens_evaluated": response.get("tokens_evaluated"),
                           "cached_tokens": cache_tokens,
                           "stop_type": response.get("stop_type"),
                           "timings": response.get("timings"), "config": cfg}
        return parsed
