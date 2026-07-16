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


def synth_prompt(n_tokens: int) -> str:
    """~1 short word per token. Words vary (w0, w1, ...) rather than repeating a single token so
    a real BPE tokenizer doesn't collapse a long identical run into fewer tokens than intended."""
    if n_tokens <= 0:
        return ""
    return " ".join(f"w{i % 4999}" for i in range(n_tokens))


def build_messages(scenario: Scenario, client_id: int = 0) -> list[dict]:
    """Build the chat messages list for one client's request under this scenario.

    shared_prefix_tokens (if any) becomes a system message with IDENTICAL text across every
    client of the scenario -- that's the point of S3: a shared, cacheable prefix. prompt_tokens
    becomes a per-client user message with a client_id-salted prefix so the text isn't
    byte-identical across clients.
    """
    messages = []
    if scenario.shared_prefix_tokens:
        messages.append({"role": "system", "content": synth_prompt(scenario.shared_prefix_tokens)})
    messages.append({
        "role": "user",
        "content": f"[client {client_id}] {synth_prompt(scenario.prompt_tokens)}",
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
