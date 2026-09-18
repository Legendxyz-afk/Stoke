"""Windows listen-backlog semantics, no engine involved: a listener with backlog B that
accepts one connection every `step` seconds, and N clients connecting at once.
Prints how many were refused, how many connected, and when.

    python tests/tooling/backlog_probe.py

Run 2026-09-18 after the @32 sitting lost 13 of 32 stoke clients to WinError 10061:
  backlog=16 n=40 step=0.07s: accepted 40, refused 0                 (last accept 2.6 s)
  backlog=16 n=32 step=2.8s:  accepted 17, refused 15 (10061 at ~2.0 s) (last accept 39 s)
  backlog=48 n=32 step=2.8s:  accepted 32, refused 0                 (last accept 81 s)
So: Windows DROPS a SYN that finds the queue full, the client retransmits at ~1 s
and ~2 s, and gives up as "actively refused" -- which is why a server draining
the queue at decode speed (70 ms turns) never shows it and one in prefill
(2.8 s chunk turns) loses everything past the backlog. The stoke server accepts
one connection per turn (server.c), hence backlog = MAX_CONNS (commit after
9eca59e), and hence why burst_smoke.py must hit the server DURING a prefill.
"""
import socket, threading, time, sys

def probe(backlog, n, step, accept_first=3):
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", 0)); ls.listen(backlog)
    port = ls.getsockname()[1]
    res = [None] * n
    t0 = time.perf_counter()
    def client(i):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(60)
        try:
            s.connect(("127.0.0.1", port)); res[i] = ("ok", time.perf_counter() - t0)
            s.sendall(b"x"); time.sleep(step * (n + 2))
        except OSError as e:
            res[i] = (f"{type(e).__name__} {getattr(e, 'winerror', '')}", time.perf_counter() - t0)
        finally:
            s.close()
    ths = [threading.Thread(target=client, args=(i,)) for i in range(n)]
    for t in ths: t.start()
    accepted = []
    ls.settimeout(step * (n + 5))
    for k in range(n):
        if k >= accept_first: time.sleep(step)
        try:
            c, _ = ls.accept(); accepted.append((k, round(time.perf_counter() - t0, 2)))
        except socket.timeout:
            break
    for t in ths: t.join()
    ls.close()
    ok = [r for r in res if r and r[0] == "ok"]
    bad = [r for r in res if r and r[0] != "ok"]
    kinds = {}
    for r in bad: kinds[r[0]] = kinds.get(r[0], 0) + 1
    print(f"backlog={backlog} n={n} step={step}s: accepted {len(accepted)}, client ok {len(ok)}, refused/err {len(bad)} {kinds}; "
          f"err times {[round(r[1],2) for r in bad][:6]}; last accept at {accepted[-1][1] if accepted else None}s")

if __name__ == "__main__":
    probe(16, 40, 0.07)   # the smoke's shape: decode-speed accepts
    probe(16, 32, 2.8)    # the sitting's shape: chunk-speed accepts
    probe(48, 32, 2.8)    # the fix
