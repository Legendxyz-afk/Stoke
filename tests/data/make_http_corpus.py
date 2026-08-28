#!/usr/bin/env python3
"""Generates tests/data/http_corpus/ -- M6 gate item [6]'s committed
garbage-input corpus for src/http.c.

Run from the repo root:  python tests/data/make_http_corpus.py

▶▶ THE FILES ARE COMMITTED, NOT GENERATED AT BUILD TIME. This script exists so
they are reviewable and reproducible (the tests/data/make_fixtures.py pattern),
not so the test suite depends on Python. tests/test_http.c reads the directory
directly.

▶▶ THE EXPECTATION LIVES IN THE FILENAME:

    <expect>[.<reason-slug>].<note>.raw

`<expect>` is `ok`, `more`, or an HTTP status code; for a status code the SECOND
dot-field is the exact HttpError.reason the parser must produce. That is not
decoration. Sixteen of these files answer 400, and a corpus asserting only the
number would stay green against a parser that rejected every one of them for one
wrong cause. Putting it in the name rather than a manifest means the expectation
cannot drift away from the bytes.

⚠ tests/data/http_corpus/*.raw is marked `-text` in .gitattributes. This repo
has core.autocrlf=true, and several fixtures below exist precisely to carry a
bare LF, a bare CR, or a CRLF split across a read boundary. Without that line
git would rewrite those bytes on checkout and the fixture would quietly become a
different test -- on a fresh clone, never in the tree that wrote it.
"""

import os
import sys

OUT = os.path.join("tests", "data", "http_corpus")

CRLF = b"\r\n"


def req(line, headers=(), body=b"", eol=CRLF):
    """Assembles a request. `headers` are raw `name: value` byte strings so a
    malformed one (no colon, space before it) can be expressed at all."""
    out = line + eol
    for h in headers:
        out += h + eol
    out += eol + body
    return out


def cl(body):
    return b"Content-Length: " + str(len(body)).encode() + b""


# ---------------------------------------------------------------------------
# The corpus. Every entry is (filename, bytes). The name states the verdict;
# the comment states why the case is worth a file.
# ---------------------------------------------------------------------------

CHAT_BODY = b'{"model":"qwen3-4b","messages":[{"role":"user","content":"hi"}]}'

FILES = [
    # ---- ok: the shapes real clients actually send ------------------------
    ("ok.health-get-curl.raw",
     req(b"GET /health HTTP/1.1",
         [b"Host: 127.0.0.1:8080", b"User-Agent: curl/8.4.0", b"Accept: */*"])),

    # The openai-python client's exact header set, which is the traffic gate
    # items [1] and [3] will drive. If this file ever stops parsing, the
    # compat gate is already lost and this says so in milliseconds.
    ("ok.openai-python-chat-post.raw",
     req(b"POST /v1/chat/completions HTTP/1.1",
         [b"Host: localhost:8080",
          b"Accept-Encoding: gzip, deflate",
          b"Connection: keep-alive",
          b"Accept: application/json",
          b"Content-Type: application/json",
          b"User-Agent: OpenAI/Python 1.40.0",
          b"X-Stainless-Lang: python",
          b"X-Stainless-Package-Version: 1.40.0",
          b"X-Stainless-Retry-Count: 0",
          b"Authorization: Bearer sk-not-a-real-key",
          cl(CHAT_BODY)],
         CHAT_BODY)),

    ("ok.models-get-with-query.raw",
     req(b"GET /v1/models?limit=5&after=x HTTP/1.1", [b"Host: localhost"])),

    ("ok.http10-connection-keep-alive.raw",
     req(b"GET /health HTTP/1.0", [b"Connection: keep-alive"])),

    # A second complete request follows the first. The parser must consume
    # exactly head_len+body_len and leave the rest for the next call --
    # swallowing it is how a pipelined client gets its second request eaten.
    ("ok.pipelined-second-request.raw",
     req(b"POST /v1/completions HTTP/1.1", [cl(b"HELLO")], b"HELLO")
     + req(b"GET /health HTTP/1.1", [b"Host: x"])),

    # Bare LF throughout: accepted deliberately (see line_end in src/http.c).
    # This is the file .gitattributes -text exists for.
    ("ok.lf-only-line-endings.raw",
     req(b"POST /v1/completions HTTP/1.1", [cl(b"hi")], b"hi", eol=b"\n")),

    ("ok.empty-header-values.raw",
     req(b"GET / HTTP/1.1", [b"X-Empty:", b"X-Ows:   ", b"Host: x"])),

    ("ok.content-length-zero.raw",
     req(b"POST /v1/completions HTTP/1.1", [b"Content-Length: 0"])),

    # Deep OWS and mixed case on names that MEAN something. A parser that
    # case-folded wrongly would mis-read Content-Length here and hang.
    ("ok.mixed-case-and-tabs.raw",
     req(b"POST /v1/completions HTTP/1.1",
         [b"cOnTeNt-LeNgTh:\t 5 ", b"CONNECTION:  Close  "], b"HELLO")),

    # ---- more: incomplete, and answered rather than hung up on ------------
    ("more.slowloris-partial-header.raw", b"GET /health HTTP/1.1\r\nHost: 12"),
    ("more.headers-without-blank-line.raw",
     b"GET / HTTP/1.1\r\nHost: x\r\nAccept: */*\r\n"),
    # Ends mid-CRLF: the classic read-boundary case.
    ("more.split-inside-crlf.raw", b"GET / HTTP/1.1\r\nHost: x\r\n\r"),
    ("more.body-short-by-one.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 10"], b"123456789")),
    ("more.request-line-only-partial.raw", b"POST /v1/chat/comp"),

    # ---- 400: malformed, one file per DISTINCT cause ----------------------
    ("400.bare-cr.in-request-line.raw", b"GET / HTTP/1.1\rX\r\n\r\n"),
    ("400.bare-cr.in-header-value.raw",
     req(b"GET / HTTP/1.1", [b"Host: a\rb: c"])),
    ("400.control-char.nul-in-target.raw",
     req(b"GET /a\x00b HTTP/1.1", [b"Host: x"])),
    ("400.control-char.vertical-tab-in-value.raw",
     req(b"GET / HTTP/1.1", [b"X-Vt: a\x0bb"])),
    ("400.control-char.del-in-header-name.raw",
     req(b"GET / HTTP/1.1", [b"X\x7fY: z"])),
    ("400.bad-request-line.double-space.raw", b"GET  / HTTP/1.1\r\n\r\n"),
    ("400.bad-request-line.no-version.raw", b"GET /health\r\n\r\n"),
    ("400.bad-request-line.leading-blank-line.raw",
     b"\r\nGET / HTTP/1.1\r\n\r\n"),
    # asterisk-form is legal HTTP (OPTIONS *) and still refused: this server
    # has four origin-form endpoints and nothing to say about the others.
    ("400.bad-target.asterisk-form.raw", b"OPTIONS * HTTP/1.1\r\n\r\n"),
    ("400.bad-target.absolute-form.raw",
     b"GET http://elsewhere.example/v1/models HTTP/1.1\r\nHost: x\r\n\r\n"),
    ("400.bad-version.no-dot.raw", b"GET / HTTP/11\r\n\r\n"),
    ("400.bad-version.protocol-name.raw", b"GET / HTTPS/1.1\r\n\r\n"),
    ("400.bad-method.brace.raw", b"GE{T / HTTP/1.1\r\n\r\n"),
    ("400.header-no-colon.raw", req(b"GET / HTTP/1.1", [b"Host"])),
    ("400.header-empty-name.raw", req(b"GET / HTTP/1.1", [b": value"])),
    ("400.bad-header-name.brace.raw", req(b"GET / HTTP/1.1", [b"X{Y: z"])),
    ("400.obs-fold.tab-continuation.raw",
     req(b"GET / HTTP/1.1", [b"Host: a", b"\tcontinued"])),
    ("400.obs-fold.space-continuation.raw",
     req(b"GET / HTTP/1.1", [b"Host: a", b"  continued"])),

    # ⚠ The Transfer-Encoding smuggling probe with whitespace before the colon.
    # It must be 400 and NOT 501: the header is malformed, so it is refused
    # before anything decides what its value means. An intermediary that
    # trimmed the space would see a TE header the origin never did, which is
    # the whole trick -- and refusing the message ends it for both.
    ("400.space-before-colon.te-obfuscation.raw",
     req(b"POST /x HTTP/1.1",
         [b"Transfer-Encoding : chunked", b"Content-Length: 6"])),

    # Differing values, the canonical CL.CL desync probe.
    ("400.dup-content-length.differing.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 6", b"Content-Length: 5"],
         b"HELLO!")),
    ("400.dup-content-length.identical.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 5", b"Content-Length: 5"],
         b"HELLO")),
    ("400.bad-content-length.hex.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 0x10"])),
    ("400.bad-content-length.negative.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: -1"])),
    ("400.bad-content-length.whitespace-inside.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 1 0"])),
    ("400.bad-content-length.overflow.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 99999999999999999999999999"])),

    # ---- 413: refused on the DECLARATION, before a body byte arrives ------
    ("413.body-too-large.one-over-cap.raw",
     req(b"POST /v1/chat/completions HTTP/1.1",
         [b"Content-Type: application/json", b"Content-Length: 1048577"])),
    ("413.body-too-large.declared-gigabyte.raw",
     req(b"POST /x HTTP/1.1", [b"Content-Length: 1073741824"])),

    # ---- 431: the two head budgets, and the slowloris shape ---------------
    ("431.head-too-large.long-cookie.raw",
     req(b"GET / HTTP/1.1", [b"Host: x", b"Cookie: " + b"a" * 9000])),
    ("431.too-many-headers.raw",
     req(b"GET / HTTP/1.1",
         [("X-%03d: v" % i).encode() for i in range(70)])),
    # No terminator at all: a client that never finishes its head must be cut
    # off at the cap rather than held on NEED_MORE forever.
    ("431.head-too-large.no-terminator.raw",
     b"GET / HTTP/1.1\r\n" + b"a" * 9000),

    # ---- 501: framing we refuse LOUDLY rather than half-honour ------------
    ("501.transfer-encoding.classic-cl-te.raw",
     req(b"POST /v1/completions HTTP/1.1",
         [b"Host: x", b"Content-Length: 6", b"Transfer-Encoding: chunked"],
         b"0\r\n\r\n")),
    ("501.transfer-encoding.te-before-cl.raw",
     req(b"POST /x HTTP/1.1",
         [b"Transfer-Encoding: chunked", b"Content-Length: 6"], b"0\r\n\r\n")),
    ("501.transfer-encoding.case-varied.raw",
     req(b"POST /x HTTP/1.1", [b"TrAnSfEr-EnCoDiNg: ChUnKeD"])),
    # Not chunked. The hazard is the DISAGREEMENT about framing, not the value,
    # so every Transfer-Encoding is refused.
    ("501.transfer-encoding.identity.raw",
     req(b"POST /x HTTP/1.1",
         [b"Transfer-Encoding: identity", b"Content-Length: 0"])),

    # ---- 505: legible, and not ours --------------------------------------
    ("505.http-version.http20.raw", b"GET / HTTP/2.0\r\nHost: x\r\n\r\n"),
    ("505.http-version.http09.raw", b"GET / HTTP/0.9\r\n\r\n"),
    ("505.http-version.minor-nine.raw", b"GET / HTTP/1.9\r\nHost: x\r\n\r\n"),
]


def main():
    if not os.path.isdir(os.path.join("tests", "data")):
        sys.exit("run me from the repo root (tests/data/ not found)")
    os.makedirs(OUT, exist_ok=True)

    names = [n for n, _ in FILES]
    if len(set(names)) != len(names):
        sys.exit("duplicate filename in FILES -- one case would silently "
                 "overwrite another")

    for name, data in FILES:
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
    print("wrote %d files to %s" % (len(FILES), OUT))


if __name__ == "__main__":
    main()
