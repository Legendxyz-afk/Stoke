#!/usr/bin/env python3
"""Does `stoke serve` hold a burst of N simultaneous connections while it is generating?

    python tests/tooling/burst_smoke.py build/win/stoke.exe --burst 40 [--port 18191]

Starts the server (--seqs 4 --ctx 2048), waits for /health, sends ONE request
with a ~1,000-token prompt in the background so the server is in PREFILL --
its turns are then whole 256-token chunk steps (~2.8 s each) and the accept
loop, one accept per turn, drains the listen queue at that pace -- then opens
`burst` TCP connections at once, each sending GET /health, and counts how many
are answered against how many are refused at connect.

⚠ The first cut of this smoke burst during DECODE (70 ms turns) and read 40/40
on the backlog-16 binary that had just lost 13 of 32 clients in the @32
sitting: Windows drops a SYN that finds the queue full, the client retransmits
at ~1 s and ~2 s, and a queue draining at decode speed has room again before
the retries run out. Only a queue that stays full for > ~2 s refuses (WinError
10061). tests/tooling/backlog_probe.py measures the semantics with no engine.
Prints one verdict line; exit 0 iff every connection was answered. Run it
against the OLD binary first -- it must go red there, or it is not testing
the backlog.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


def one(port, results, i):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(120)  # a queued connection is answered one server turn at a time
    try:
        s.connect(("127.0.0.1", port))
        s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data and (b"Content-Length" in data or data.endswith(b"}")):
                break
        results[i] = "ok" if data.startswith(b"HTTP/1.1 200") else f"bad: {data[:40]!r}"
    except ConnectionRefusedError:
        results[i] = "refused"
    except OSError as e:
        results[i] = f"err: {e}"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("--burst", type=int, default=30)  # < MAX_CONNS - 1 (the generating request holds one)
    ap.add_argument("--port", type=int, default=18191)
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    # --ctx must HOLD the long prompt (a prompt past ctx is refused at once and
    # the server never enters prefill -- the second blind cut of this smoke);
    # --seqs is not the connection table, so 4 keeps the pool small.
    srv = subprocess.Popen([exe, "serve", os.path.abspath(MODEL), "--port", str(a.port), "--seqs", "4", "--ctx", "2048",
                            "--threads", "12", "--max-seconds", "240"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1).read()
                break
            except Exception:
                time.sleep(0.2)
        else:
            print("VERDICT: server never became healthy")
            return 2
        # one request with a long prompt, so the server sits in prefill (2.8 s
        # chunk turns) while the burst arrives; ~1,000 tokens = four chunks.
        prompt = ("The river rises in the hills, gathers the rain, and carries it past the towns to the sea. " * 60)
        body = json.dumps({"model": "stoke", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 16, "stream": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})

        def generate():
            try:
                urllib.request.urlopen(req, timeout=300).read()
            except Exception:
                pass  # the server is killed under it at the end; not the verdict
        gen = threading.Thread(target=generate, daemon=True)
        gen.start()
        time.sleep(1.5)  # into the first prefill chunk
        results = [None] * a.burst
        ts = [threading.Thread(target=one, args=(a.port, results, i)) for i in range(a.burst)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        dt = time.time() - t0
        ok = sum(1 for r in results if r == "ok")
        refused = sum(1 for r in results if r == "refused")
        other = [r for r in results if r not in ("ok", "refused")]
        print(f"VERDICT: burst {a.burst} while generating -> answered {ok}, refused {refused}, other {len(other)} "
              f"in {dt:.1f}s  [{os.path.basename(os.path.dirname(a.exe))}/{os.path.basename(a.exe)}]"
              + (f"  other={other[:3]}" if other else ""))
        return 0 if ok == a.burst else 1
    finally:
        srv.kill()
        srv.wait()


if __name__ == "__main__":
    sys.exit(main())
