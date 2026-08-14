# M4.5 -- the NULL-1 base-rate run. What does the 50-prompt first-divergence
# instrument report on Q4_K_M for a change with ZERO semantic content?
#
# Side A = build/stoke-null0.exe (ed2aa32 rebuilt, stream-identical to the m4
# gate archive). Side B = build/stoke-null1.exe (the same source with the eight
# f32 lane additions reversed). Both --kv f32. The ONLY difference between the
# two sides is f32 addition order.
#
# The reading is FIXED IN ADVANCE:
#   docs/superpowers/design-notes/2026-08-14-m45-null1-baserate-preregistration.md
# Read it before looking at the number.
#
# Same harness, corpus, -n and probe-spec as the C grading run, so the two
# results are comparable. tau is frozen from THIS pair's own agreeing positions
# before it grades anything.
#
# Resumable: every probe pair, sweep row and verdict is fsync'd as it lands.
# ExecutionTimeLimit must be PT0S -- it is a second deadline that appears in no
# log and kills mid-leg.

$ErrorActionPreference = 'Stop'
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFF   # P-cores, standing call

$root = "M:\New Drive\Desktop\Programming\stoke"
$py   = "C:\Users\Kabir\AppData\Local\Programs\Python\Python314\python.exe"
$out  = "$root\docs\superpowers\gate-reports\2026-08-14-m45-null1-baserate"
$log  = "$out-raw.txt"

$argv = @(
    "-u", "$root\tests\tooling\m4_q8kv_gate.py",
    "--stoke",   "$root\build\stoke-null0.exe",
    "--stoke-b", "$root\build\stoke-null1.exe",
    "--kv-a", "f32", "--kv-b", "f32",
    "--model", "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "-n", "64", "--corpus", "50", "--threads", "6",
    "--out", $out
)
if (Test-Path "$out-run.json") { $argv += "--resume" }

"=== {0} launch: {1}" -f (Get-Date -Format s), ($argv -join ' ') |
    Out-File -Append -Encoding utf8 $log

& $py @argv 2>&1 | Out-File -Append -Encoding utf8 $log
$code = $LASTEXITCODE      # captured immediately; an echo here would mask it

"=== {0} python exit={1}" -f (Get-Date -Format s), $code |
    Out-File -Append -Encoding utf8 $log
exit $code
