#!/usr/bin/env python3
"""One-off diagnostic: dump every raw SSE `data:` line an engine sends for one tiny request.
Not part of the harness -- exists to settle chunk-anatomy questions empirically (M0 Task 2).
Usage: python bench/probe_sse.py [ollama|llamacpp] [max_tokens]
"""
import http.client
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import adapters  # noqa: E402


def dump(base_url, model, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the word hello and nothing else."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    host, port = base_url.split("//", 1)[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=300)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(payload).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    print(f"HTTP {resp.status}")
    n = 0
    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            continue
        n += 1
        print(f"[{n:02d}] {line}")
        if line == "data: [DONE]":
            break
    conn.close()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which == "ollama":
        adapter = adapters.OllamaAdapter()
    else:
        adapter = adapters.LlamaCppAdapter()
    try:
        adapter.start()
        if which == "ollama":
            adapters.ensure_ollama_model()
        dump(adapter.base_url, adapter.model_name, max_tokens)
    finally:
        adapter.stop()


if __name__ == "__main__":
    main()
