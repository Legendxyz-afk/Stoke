# M2 parity corpus — MANIFEST

Corpus (`p00`–`p49`, 50 prompts) + reserve pool (`r00`–`r09`, 10 prompts) for
the M2 greedy-parity gate. Composition per
`docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md`
§8. Token counts are the REAL tokenizer output of

```
build/win/stoke.exe tokenize models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf <file>
```

(count = number of ids in the printed `[id, id, …]` line). Every file is
valid UTF-8, has no BOM, and its last byte is not `0x0A` (no trailing
newline) — verified mechanically for all 60 files, see the report at
`.superpowers/sdd/m2-corpus-report.md`.

**Trailing-whitespace exception (per brief §"Hard constraints"):** two files
deliberately end in a run of space characters (`0x20`), not a newline, as
part of the adversarial "trailing whitespace runs" design. This still
satisfies `last byte != 0x0A`, but is called out here as instructed:
- `p46.txt` — ends in 4 trailing spaces after "...and".
- `r09.txt` — ends in 6 trailing spaces after "...sure?!?!".

## Corpus (`tests/data/parity/p00.txt` … `p49.txt`)

| id | category | tokens | description |
|---|---|---|---|
| p00 | prose | 23 | Explain photosynthesis from sunlight absorption to glucose |
| p01 | prose | 25 | Describe a city skyline at night from rooftop |
| p02 | prose | 23 | Lighthouse keeper's last climb up the spiral stairs |
| p03 | prose | 26 | Explain pour-over coffee brewing steps and temperatures |
| p04 | prose | 18 | Explain the difference between a comet and asteroid |
| p05 | prose | 21 | Woman opens a letter, uncertain what news it holds |
| p06 | prose | 24 | Summarize a locked-room murder mystery novel's plot |
| p07 | prose | 27 | Explain why the sky is blue, sunset orange/red |
| p08 | prose | 21 | Give three tips for staying focused while studying |
| p09 | prose | 21 | Train departs station as first winter snow falls |
| p10 | prose | 24 | Explain hurricane vs typhoon vs cyclone naming differences |
| p11 | prose | 22 | Describe a bakery's smells and sounds before sunrise |
| p12 | prose | 22 | Explain how a moving bicycle stays balanced upright |
| p13 | prose | 26 | Familiar forest trail feels unusually quiet and different |
| p14 | prose | 22 | Explain the water cycle simply to a child |
| p15 | code | 40 | Complete C function reversing a singly linked list |
| p16 | code | 22 | Complete Python Fibonacci function using memoization caching |
| p17 | code | 47 | Complete JSON web server config, explain each field |
| p18 | code | 49 | Complete C Vec3 struct and magnitude function |
| p19 | code | 41 | Complete Python Stack class push/pop/peek methods |
| p20 | code | 64 | Explain C summing loop code, note potential bugs |
| p21 | code | 63 | Explain Python function filtering multiples of three/five |
| p22 | code | 88 | Explain each field in a JSON user account object |
| p23 | code | 39 | Explain C double-free bug, undefined behavior |
| p24 | code | 51 | Trace recursive Python factorial-style function with n=4 |
| p25 | multilingual | 16 | Chinese: explain artificial intelligence and its daily impact |
| p26 | multilingual | 19 | Japanese: explain characteristics of Japan's four seasons |
| p27 | multilingual | 26 | Korean: three daily habits for healthy living |
| p28 | multilingual | 82 | Hindi (Devanagari): best methods to concentrate while studying |
| p29 | multilingual | 24 | Arabic: explain how the four seasons change |
| p30 | multilingual | 53 | Mixed-emoji English trip recap, continue the story |
| p31 | multilingual | 25 | Chinese: translate an idiom, explain cultural background |
| p32 | multilingual | 104 | Hindi (Devanagari): old farmer village story, continue |
| p33 | multilingual | 36 | Arabic: old fisherman village story, continue narrative |
| p34 | multilingual | 34 | Korean: quiet winter night snow, continue story |
| p35 | math | 53 | Word problem: two trains meeting, distance and time |
| p36 | math | 45 | Simplify and factor a quadratic algebraic expression |
| p37 | math | 26 | Convert miles per hour to meters per second |
| p38 | math | 36 | Solve linear equation for x, show each step |
| p39 | math | 31 | Rectangular garden perimeter word problem, find dimensions |
| p40 | long-context | 490 | History passage: printing press effects; summarize in 3-4 sentences |
| p41 | long-context | 477 | Science passage: coral reef biology and types; continue re: threats |
| p42 | long-context | 462 | Narrative passage: mountain trail journey; summarize the trip |
| p43 | long-context | 471 | Tech-history passage: early internet/ARPANET; continue the story |
| p44 | long-context | 447 | Fiction passage: lighthouse keeper mystery; continue the story |
| p45 | adversarial | 14 | Leading whitespace run (10 spaces) before real instruction |
| p46 | adversarial | 22 | Trailing whitespace run — see exception note above |
| p47 | adversarial | 26 | Mixed-script mid-word invented term ("dragon龍fire") question |
| p48 | adversarial | 2 | Near-empty prompt: "Why?" (2 tokens) |
| p49 | adversarial | 18 | Repeated punctuation runs ("...", "?!?!", "!!!"), excited demand |

### Corpus totals

| category | files | target | token range |
|---|---|---|---|
| prose/instruction | 15 (p00–p14) | ~15 | 18–27 |
| code | 10 (p15–p24) | ~10 | 22–88 |
| multilingual | 10 (p25–p34) | ~10 | 16–104 |
| math/symbolic | 5 (p35–p39) | ~5 | 26–53 |
| long-context | 5 (p40–p44) | ~5 | 447–490 (all within 400–512) |
| adversarial-ish | 5 (p45–p49) | ~5 | 2–26 |
| **total** | **50** | **50** | 2–490 |

Multilingual scripts covered (≥5 required): Chinese (Han), Japanese
(hiragana/katakana/kanji), Korean (Hangul), Devanagari (Hindi), Arabic, and
Latin+emoji — 6 flavors across p25–p34.

## Reserve pool (`tests/data/parity/r00.txt` … `r09.txt`)

| id | category | tokens | description |
|---|---|---|---|
| r00 | prose | 20 | What makes a detective story compelling, two techniques |
| r01 | prose | 25 | Clock tower chimes again after twenty silent years |
| r02 | code | 22 | Complete C recursive factorial function |
| r03 | code | 49 | Explain Python median-via-sort function on a 5-item list |
| r04 | multilingual | 16 | Chinese: importance of sleeping on time, two tips |
| r05 | multilingual | 22 | Arabic: three practical tips for learning a language |
| r06 | math | 57 | Rectangular pool volume and fill-time word problem |
| r07 | long-context | 472 | Science passage: the water cycle; summarize in 2-3 sentences |
| r08 | adversarial | 2 | Near-empty prompt: "Hi." (2 tokens) |
| r09 | adversarial | 9 | Leading/trailing whitespace + repeated punctuation — see exception note |

### Reserve totals

| category | files | target | token range |
|---|---|---|---|
| prose/instruction | 2 (r00–r01) | 2 | 20–25 |
| code | 2 (r02–r03) | 2 | 22–49 |
| multilingual | 2 (r04–r05) | 2 | 16–22 |
| math/symbolic | 1 (r06) | 1 | 57 |
| long-context | 1 (r07) | 1 | 472 (within 400–512) |
| adversarial-ish | 2 (r08–r09) | 2 | 2–9 |
| **total** | **10** | **10** | 2–472 |

## Grand total

50 corpus + 10 reserve = **60 files**, all under `tests/data/parity/`, all
mechanically verified (real tokenizer counts, UTF-8, no BOM, no trailing
`0x0A`). No file contains any special-token string (`<|im_start|>` etc.) —
raw mode plus specials is M1-tested territory, not this corpus's job.
