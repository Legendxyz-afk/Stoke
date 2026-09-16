"""Scenario definitions for the stoke bench harness (M0 Task 1).

Prompt text is synthesized as repeated word chunks sized to roughly hit the requested token
counts. Exact tokenization doesn't matter for mock_server.py (it never reads message content,
only the mock_p_ms/mock_d_ms/mock_t control knobs riding at the top level of the request body);
real engines (Task 2+) receive this same synthesized text as plain user/system messages and will
tokenize it for real, so "roughly hit the token counts" is all that's required here.
"""
from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    n_clients: int
    prompt_tokens: int
    max_tokens: int
    shared_prefix_tokens: int
    mode: str  # "closed_loop" | "poisson"
    arrival_rate: float | None = None  # events/sec, required iff mode == "poisson"
    requests_per_client: int = 1  # Task 1 scope is exactly 1; >1 is a Task 2 extension point


# 135 common English words, each verified to tokenize to EXACTLY 1 token (with its leading
# space; first-word-no-space case also 1) in Qwen3-4B-Instruct-2507's BPE vocab, via
# bench/vendor/llama-b10037/llama-tokenize.exe on 2026-07-16: 135 words -> 135 token ids.
_WORDS = (
    "the of and to in is it for on as at by an be or we he was are this that with from have "
    "has had not but all can will one two three four five six seven eight nine ten out up "
    "down over under about into after before between through during each some many most "
    "other more less very just also then than when where which while who what why how do "
    "does did done make made take took give gave find found think thought know knew see saw "
    "get got go went come came say said work word time year day way man men new old good "
    "great small large long short high low right left part place case fact hand eye life "
    "world house point government company number group problem question"
).split()


def synth_prompt(n_tokens: int) -> str:
    """~1 real token per word: cycles _WORDS, whose members are calibrated to be exactly 1 BPE
    token each for the reference model (see _WORDS comment). BPE merges never cross the
    space-delimited pre-token boundary, so a cycled pool stays 1 token/word regardless of
    repetition. History: the original scheme emitted synthetic words like 'w417' that split into
    3-4 BPE tokens each, inflating a nominal 512-token prompt to ~1950 real tokens -- caught in
    M0 Task 3's real mini-cell when the tuned per-slot ctx=2048 silently truncated generation at
    94 of 128 requested tokens. Nominal-vs-real drift is now only the chat template plus any
    salt/client prefix (~10 tokens), re-verified against llama-tokenize whenever _WORDS changes."""
    if n_tokens <= 0:
        return ""
    return " ".join(_WORDS[i % len(_WORDS)] for i in range(n_tokens))


def build_messages(scenario: Scenario, client_id: int = 0, salt: str | None = None) -> list[dict]:
    """Build the chat messages list for one client's request under this scenario.

    shared_prefix_tokens (if any) becomes a system message with IDENTICAL text across every
    client of the scenario -- that's the point of S3: a shared, cacheable prefix. prompt_tokens
    becomes a per-client user message with a client_id-salted prefix so the text isn't
    byte-identical across clients.

    salt (M0 Task 3 addition): identical prompts across measured runs let engines with
    prefix/prompt caching skip prefill on runs 2-3, silently inflating them; a per-run salt
    defeats cross-run cache reuse while keeping S3's shared system prefix identical across
    clients *within* a run (that within-run sharing is the capability S3 measures).
    salt=None (default) reproduces today's output byte-for-byte. When salt is set, the system
    message becomes f"[run {salt}] {<same synth text>}" and the user message becomes
    f"[run {salt} client {client_id}] {<same synth text>}".
    """
    system_text = synth_prompt(scenario.shared_prefix_tokens)
    user_text = synth_prompt(scenario.prompt_tokens)
    messages = []
    if salt is None:
        if scenario.shared_prefix_tokens:
            messages.append({"role": "system", "content": system_text})
        messages.append({
            "role": "user",
            "content": f"[client {client_id}] {user_text}",
        })
    else:
        if scenario.shared_prefix_tokens:
            messages.append({"role": "system", "content": f"[run {salt}] {system_text}"})
        messages.append({
            "role": "user",
            "content": f"[run {salt} client {client_id}] {user_text}",
        })
    return messages


# Locked scenario instances (spec: docs/superpowers/plans/2026-07-16-phase1-implementation.md, Task 1)

S1 = Scenario(
    name="s1-interactive",
    n_clients=8,
    prompt_tokens=512,
    max_tokens=128,
    shared_prefix_tokens=0,
    mode="poisson",
    arrival_rate=0.5,
)

S2 = [
    Scenario(
        name=f"s2-closed-{n}",
        n_clients=n,
        prompt_tokens=512,
        max_tokens=128,
        shared_prefix_tokens=0,
        mode="closed_loop",
    )
    for n in (1, 2, 4, 8, 16)
]

# 2026-09-16: one more S2 width for the "N > 16" question (neither engine is
# measured past 16). Same shape, NOT a member of the locked family the M0 bars
# were drawn on, and NOT in run_matrix's default set -- selectable by name only,
# the way stoke-* configs are, so a default full-matrix run does not silently
# grow a 32-client cell against every baseline.
S2_WIDE = [
    Scenario(
        name="s2-closed-32",
        n_clients=32,
        prompt_tokens=512,
        max_tokens=128,
        shared_prefix_tokens=0,
        mode="closed_loop",
    )
]

S3 = Scenario(
    name="s3-agent-fanout",
    n_clients=8,
    prompt_tokens=200,
    max_tokens=300,
    shared_prefix_tokens=1024,
    mode="closed_loop",
)

S4 = Scenario(
    name="s4-longctx",
    n_clients=1,
    prompt_tokens=4096,
    max_tokens=128,
    shared_prefix_tokens=0,
    mode="closed_loop",
)
