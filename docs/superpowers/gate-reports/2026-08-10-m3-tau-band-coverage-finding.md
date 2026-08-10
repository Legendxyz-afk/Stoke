# [2v2-iii] τ_Q4KM band coverage — the continuations are load-bearing, not bookkeeping

Date: 2026-08-10, written mid-run with RUN A at 5/6 legs banked. Author: Opus 5.

**Finding in one line: band [100,300) has ZERO agreeing positions before the
required continuations run, and band [300,600) depends entirely on p41 agreeing
through index 19.** Both are computed from measured divergence indexes, not
projected.

This is a coverage fact about the criterion, not a re-reading of it. Nothing
here changes ruling §5, freezes anything, or grades anything.

## The arithmetic

Absolute position = prompt_len + generation index (`ref_forward.py` records
generation-relative indexes; the 2026-08-07 band correction). Prompt lengths
are the driver's own printed counts.

| prompt | tokens | absolute range | band [0,100) | band [100,300) |
| --- | --- | --- | --- | --- |
| p00 | 23 | 23–86 | 64 | 0 |
| p10 | 24 | 24–87 | 64 | 0 |
| p20 | 64 | 64–127 | 36 | **28** |
| p30 | 53 | 53–116 | 47 | **17** |
| p45 | 14 | 14–77 | 64 | 0 |
| p41 | 477 | 477–540 | 0 | 0 (band [300,600): **64**) |

Totals: [0,100) = 275, [100,300) = 45, [300,600) = 64. Sum 384, matching the
grid's covered-position count.

τ freezes from **20 agreeing positions per occupied band**.

## Why bands 2 and 3 are exposed and band 1 is not

Band [0,100) has 275 positions against a requirement of 20 — ample, and every
prompt contributes to it from its first generated token.

**Band [100,300) is reachable only in the TAILS of p20 and p30**: p20's
band-2 positions are generation indexes 36–63, p30's are 47–63. Measured first
divergences are **p20 at index 21** and **p30 at index 18** — both far short of
where their band-2 coverage begins. Every one of those 45 positions therefore
sits past a divergence, so the band contributes **zero** agreeing positions
until a continuation re-locks the stream.

**Band [300,600) is p41 alone**, 64 positions, generation indexes 0–63. Twenty
agreeing positions requires p41 to agree through index 19. The four measured
divergences so far are at indexes **9, 18, 19, 21** — so whether p41 clears
index 19 on its first pass is close to a coin flip.

## What follows

1. **The required continuations are load-bearing for τ_2 and τ_3.** §5 already
   requires them so grid coverage returns to ~384 and the tripwire denominator
   holds; this note records a second, harder dependency — without them two of
   the three bands cannot freeze a τ at all.
2. **§5 authorizes exactly ONE continuation per first divergence.** A
   continuation that diverges again *before reaching its band* leaves that band
   short of 20 and therefore ungradeable. For p20 the continuation must survive
   from its divergence to generation index 36; for p30, to index 47.
3. **Sequencing consequence:** p41's continuation, if it diverges, is a second
   ~541-position leg. Band 3 has no other contributor, so it cannot be covered
   from anywhere else.

## What this note deliberately does not do

It does not look at any logit value, top-5 set or margin. Ruling §2.3 requires
τ to be frozen from data before that data grades anything, and inspecting
divergence magnitudes now — while τ is unfrozen — is the ordering error that
rule exists to prevent. Only the divergence INDEXES are used here, which the
driver already logs as a matter of course.

Should a band prove unable to reach 20 agreeing positions after its authorized
continuation, that is a criterion question (how to grade positions in a band
with no frozen τ) and therefore §8.4-reserved: under the 2026-08-07 governance
change Opus decides, records the reasoning with a direction check, and Kabir
overrules. It is not decided here, and it may not arise.
