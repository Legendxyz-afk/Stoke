#!/usr/bin/env python3
"""Stdlib-only mock OpenAI-compatible chat-completions SSE server (M0 Task 1).

POST /v1/chat/completions reads mock_p_ms/mock_d_ms/mock_t (defaults 200/50/20) off the top
level of the JSON body, sleeps mock_p_ms (simulated prefill), then emits mock_t SSE chunks at
exactly mock_d_ms intervals, then a terminating `data: [DONE]`. Every request also gets its own
Transfer-Encoding: chunked HTTP framing (independent of the SSE-level "data: ..." framing above).

Scheduling uses absolute deadlines (t0 + p + k*d) checked against time.perf_counter() before each
sleep, rather than mock_t successive `time.sleep(d)` calls -- the latter accumulates the overhead
of each loop iteration (JSON encode, socket write, scheduler wake latency) onto the next sleep,
drifting later every chunk. Recomputing the absolute deadline every iteration means only the
final chunk in a run can ever be late relative to its own deadline; earlier chunks aren't affected
by later ones' overhead.

Usage: python bench/mock_server.py --port 8100   (--port 0 asks the OS for a free port; the
bound port is always printed to stdout as "mock_server listening on http://127.0.0.1:<port>").
"""
import argparse
import json
import time
from http import server


class Handler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep stdout limited to the one startup banner line (test_harness.py parses it)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        p_ms = body.get("mock_p_ms", 200)
        d_ms = body.get("mock_d_ms", 50)
        n_tokens = body.get("mock_t", 20)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        t0 = time.perf_counter()
        p_s = p_ms / 1000.0
        d_s = d_ms / 1000.0
        token_chunk = "data: " + json.dumps({"choices": [{"delta": {"content": "tok"}}]}) + "\n\n"

        try:
            for k in range(n_tokens):
                deadline = t0 + p_s + k * d_s
                now = time.perf_counter()
                if deadline > now:
                    time.sleep(deadline - now)
                self._send_sse_chunk(token_chunk)

            self._send_sse_chunk("data: [DONE]\n\n")
            self._end_chunked()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass  # client went away mid-stream; nothing left to clean up beyond returning

    def _send_sse_chunk(self, text):
        data = text.encode("utf-8")
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _end_chunked(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class Server(server.ThreadingHTTPServer):
    daemon_threads = True  # don't let a stray handler thread block process exit


def main():
    parser = argparse.ArgumentParser(description="Mock SSE chat-completions server for bench timing validation")
    parser.add_argument("--port", type=int, default=8100, help="listen port (0 = OS-assigned free port)")
    args = parser.parse_args()

    httpd = Server(("127.0.0.1", args.port), Handler)
    bound_port = httpd.server_address[1]
    print(f"mock_server listening on http://127.0.0.1:{bound_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
