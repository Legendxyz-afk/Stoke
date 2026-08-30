#!/usr/bin/env python3
"""M6 GATE ITEM [1]: the openai-python suite -- chat, completions, models, and the sampler.

▶▶ WHY A REAL CLIENT AND NOT MORE curl. tests/serve_golden.sh already pins every byte of every
response, and it would pass against a server whose JSON is subtly wrong in a way only a typed
client notices: a missing field the SDK's model requires, a number where a string belongs, a
`choices` entry that does not validate. openai-python parses each response into a pydantic model
and raises on anything that does not fit, so this suite is the one test here that asks "would the
ecosystem's own client accept this?" rather than "do the bytes match what we recorded?".

⚠ max_retries=0, DELIBERATELY. The SDK retries by default, which would turn a 400 into several
requests, break the server's --max-requests count, and -- worse -- quietly make a flaky failure
look like a pass. Every request here is exactly one request.

⚠ THE ASSERTIONS ARE ON PARSED FIELDS, not on text the model produced. Nothing below asserts WHAT
the model said: that is the golden gate's job and it is pinned there. This asks only that the
shape is one the client can use -- with one exception, section 7, which compares answers to EACH
OTHER. Sampling cannot be checked any other way: its output is text either way, so the only
observable difference between a live sampler and a parameter dropped on the floor is whether two
answers that must differ do.

Usage: serve_openai.py <base_url>        (e.g. http://127.0.0.1:18143/v1)
"""
import sys

import openai


def fail(msg):
    print(f"serve_openai: FAIL -- {msg}", file=sys.stderr)
    sys.exit(1)


def main(argv):
    if len(argv) < 2:
        print("usage: serve_openai.py <base_url>", file=sys.stderr)
        return 2
    base_url = argv[1]

    # The server never reads Authorization -- it is a local engine, not a multi-tenant service --
    # but the SDK refuses to construct a client without a key, so one is supplied and ignored.
    client = openai.OpenAI(base_url=base_url, api_key="not-used", max_retries=0, timeout=300.0)
    print(f"serve_openai: openai-python {openai.__version__} against {base_url}", flush=True)

    # -- 1. models -------------------------------------------------------------------------------
    models = client.models.list()
    ids = [m.id for m in models.data]
    if not ids:
        fail("models.list() returned no models")
    if not all(isinstance(i, str) and i for i in ids):
        fail(f"models.list() returned a non-string or empty id: {ids!r}")
    print(f"serve_openai: models.list() -> {ids}", flush=True)
    model = ids[0]

    # -- 2. chat, non-streamed -------------------------------------------------------------------
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Name one color."}],
        max_tokens=16,
        temperature=0.0,
    )
    ch = r.choices[0]
    if ch.message.role != "assistant":
        fail(f"non-streamed chat: role is {ch.message.role!r}, expected 'assistant'")
    if not ch.message.content:
        fail("non-streamed chat: empty content")
    if ch.finish_reason not in ("stop", "length"):
        fail(f"non-streamed chat: finish_reason {ch.finish_reason!r}")
    # usage is unconditional on a non-streamed response -- that is the OpenAI contract and the
    # M0 harness's measurement mode depends on the streamed equivalent of it.
    if r.usage is None or r.usage.completion_tokens < 1:
        fail(f"non-streamed chat: usage missing or empty: {r.usage!r}")
    print(f"serve_openai: chat non-stream -> {ch.message.content!r} "
          f"finish={ch.finish_reason} usage={r.usage.completion_tokens}", flush=True)

    # -- 3. chat, streamed -----------------------------------------------------------------------
    # The traps, checked on the CHUNKS the client actually yields: role on the first delta and no
    # other, finish_reason null until the final one.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in five words."}],
        max_tokens=24,
        temperature=0.0,
        stream=True,
    )
    text, roles, finishes, n_chunks = "", 0, [], 0
    for i, chunk in enumerate(stream):
        n_chunks += 1
        d = chunk.choices[0].delta
        if d.role is not None:
            roles += 1
            if i != 0:
                fail(f"streamed chat: role arrived on chunk {i}, not the first")
        if d.content:
            text += d.content
        if chunk.choices[0].finish_reason is not None:
            finishes.append((i, chunk.choices[0].finish_reason))
    if roles != 1:
        fail(f"streamed chat: role appeared on {roles} chunks, expected exactly 1")
    if len(finishes) != 1:
        fail(f"streamed chat: finish_reason appeared {len(finishes)} times: {finishes}")
    if finishes[0][0] != n_chunks - 1:
        fail(f"streamed chat: finish_reason on chunk {finishes[0][0]} of {n_chunks}, not the last")
    if not text:
        fail("streamed chat: no content accumulated")
    print(f"serve_openai: chat stream -> {text!r} in {n_chunks} chunks, "
          f"finish={finishes[0][1]}", flush=True)

    # -- 4. streamed chat with usage -------------------------------------------------------------
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Name one color."}],
        max_tokens=16,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    usage, saw_finish_before_usage = None, False
    for chunk in stream:
        if chunk.choices and chunk.choices[0].finish_reason is not None:
            saw_finish_before_usage = True
        if chunk.usage is not None:
            usage = chunk.usage
            if not saw_finish_before_usage:
                fail("streamed usage: the usage chunk arrived BEFORE the finish chunk -- a client "
                     "that stops reading at finish_reason would still have to see it")
            if chunk.choices:
                fail(f"streamed usage: the usage chunk carried {len(chunk.choices)} choices, "
                     f"expected an empty array")
    if usage is None or usage.completion_tokens < 1:
        fail(f"streamed usage: no usage chunk under stream_options.include_usage: {usage!r}")
    print(f"serve_openai: chat stream + usage -> completion_tokens={usage.completion_tokens}",
          flush=True)

    # -- 5. completions, non-streamed ------------------------------------------------------------
    r = client.completions.create(
        model=model, prompt="The capital of France is", max_tokens=8, temperature=0.0,
    )
    if not r.choices or not r.choices[0].text:
        fail(f"completions: empty text: {r!r}")
    if r.usage is None or r.usage.completion_tokens < 1:
        fail(f"completions: usage missing: {r.usage!r}")
    print(f"serve_openai: completions -> {r.choices[0].text!r} "
          f"usage={r.usage.completion_tokens}", flush=True)

    # -- 6. completions, streamed ----------------------------------------------------------------
    stream = client.completions.create(
        model=model, prompt="The capital of France is", max_tokens=8, temperature=0.0, stream=True,
    )
    text, n_chunks = "", 0
    for chunk in stream:
        n_chunks += 1
        if chunk.choices and chunk.choices[0].text:
            text += chunk.choices[0].text
    if not text:
        fail(f"streamed completions: no text accumulated over {n_chunks} chunks")
    print(f"serve_openai: completions stream -> {text!r} in {n_chunks} chunks", flush=True)

    # -- 7. SAMPLING IS LIVE, AND THE SEED IS A PROMISE ------------------------------------------
    #
    # ▶▶ THIS IS THE ONE BLOCK HERE THAT COMPARES ANSWERS TO EACH OTHER, and it has to: every way
    # the sampling wiring can be wrong produces FLUENT TEXT. A server that parses `temperature` and
    # then drops it on the floor answers greedily, and greedy answers look perfect -- they are
    # merely the same answer forever. Nothing below asserts WHAT the model said; each assertion
    # asks whether two requests that must differ do, or whether two that must agree do.
    #
    # ⚠⚠ EACH ASSERTION IS PAIRED WITH THE ONE THAT MAKES IT NON-VACUOUS, because on its own every
    # one of them can be satisfied by a server that samples nothing at all:
    #   - "the same seed reproduces" is free for a greedy server. Its control is "a different seed
    #     differs", which a greedy server cannot pass.
    #   - "top_k 1 collapses to the argmax" is free for a server that dropped `temperature`. Its
    #     control is "a sampled answer differs from the greedy one", above it.
    # Neither half is dropped and neither is trusted alone. See the project's standing lesson that
    # a fixture can be blind to the defect it was written for.
    prompt = "Write one sentence about the sea."

    def sampled(what, **kw):
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=16, **kw)
        text = r.choices[0].message.content
        if not text:
            fail(f"sampling ({what}): empty content")
        return text

    seed = 20260830
    greedy = sampled("greedy baseline", temperature=0.0)
    wild_a = sampled("temperature 1.4", temperature=1.4, seed=seed)
    wild_b = sampled("temperature 1.4, same seed", temperature=1.4, seed=seed)
    wild_c = sampled("temperature 1.4, other seed", temperature=1.4, seed=seed + 1)

    if wild_a == greedy:
        fail(f"temperature 1.4 returned the greedy answer exactly ({greedy!r}) -- the sampler is "
             "not being reached, which is what a server that accepts the parameter and ignores it "
             "looks like")
    if wild_a != wild_b:
        fail(f"the same seed gave two different answers ({wild_a!r} then {wild_b!r}) -- a seed "
             "that does not reproduce is not a seed")
    if wild_a == wild_c:
        fail(f"two different seeds gave the same answer ({wild_a!r}) -- the seed is not reaching "
             "the sampler, and the reproduction above proves nothing without this")

    # top_k / top_p at the values where truncation leaves exactly one candidate. The draw then has
    # nowhere to land but the argmax, so a HIGH temperature must still reproduce the greedy answer
    # -- and it can only do that if the truncation arrived at the engine.
    k1 = sampled("top_k 1", temperature=1.4, seed=seed, extra_body={"top_k": 1})
    if k1 != greedy:
        fail(f"top_k 1 leaves one candidate, so any temperature must still give the greedy answer: "
             f"got {k1!r}, greedy was {greedy!r}")
    p_tiny = sampled("top_p 0.0001", temperature=1.4, seed=seed, top_p=0.0001)
    if p_tiny != greedy:
        fail("a nucleus smaller than the top token's own share keeps exactly that token, so this "
             f"must give the greedy answer too: got {p_tiny!r}, greedy was {greedy!r}")
    print(f"serve_openai: sampling -> greedy {greedy!r} / sampled {wild_a!r}", flush=True)

    print("serve_openai: OK (models, chat, chat+stream, chat+stream+usage, completions, "
          "completions+stream, sampling)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
