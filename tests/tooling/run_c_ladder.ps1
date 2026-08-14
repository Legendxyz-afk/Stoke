# M4.5 rung C -- the L2/L3 ladder for all ten ESCALATEs (ruling section 6).
#
# Side A = the m4 gate archive (pre-CONTRACT-v2). Side B = the archived HEAD
# binary that was actually graded (sha bc9dd3d6...), NOT build/win/stoke.exe --
# a rebuild from the unchanged tree produced a different sha on 2026-08-14, so
# the graded artifact is the one that must be adjudicated. Both at --kv f32.
#
# Two legs, in order:
#   1. Teacher-force a prompt for each divergent position. The script verifies
#      the round-trip id-for-id and REFUSES a position that does not match, so
#      a silent "dumped the wrong forward pass" is not possible.
#   2. Dump 36 layers per side per position and hand each pair to l2_diff.py,
#      whose STEP leg carries the verdict. Three CONTROL positions (prompts
#      where the two binaries produced identical streams) calibrate the
#      absolute leg, because l2_diff's 1e-3 is a C-vs-numpy constant that any
#      pair differing by construction blows past regardless of health.
#
# Deterministic: contention changes how long this takes and nothing else. Safe
# to stop and re-run -- the teacher-forced prompts are rebuilt idempotently and
# the dumps are overwritten.
#
# ExecutionTimeLimit must be PT0S on the task: it is a SECOND deadline that
# appears in no log and kills mid-leg.

$ErrorActionPreference = 'Stop'
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFF   # P-cores, standing call

$root = "M:\New Drive\Desktop\Programming\stoke"
$py   = "C:\Users\Kabir\AppData\Local\Programs\Python\Python314\python.exe"
$wd   = "$root\build\ladder-c"
$out  = "$root\docs\superpowers\gate-reports\2026-08-14-m45-c-ladder.txt"
$log  = "$root\docs\superpowers\gate-reports\2026-08-14-m45-c-ladder-raw.txt"

$A = "$root\build\stoke-m4-gate-f4f9b667.exe"
$B = "$root\build\stoke-m45-c-head-bc9dd3d6.exe"

# prompt:gp for every ESCALATE, read from the banked verdict -- not retyped
# from a summary. The controls are three prompts the graded sweep recorded as
# byte-identical between the two binaries.
$targets  = @("p08:54","p11:12","p18:24","p23:1","p25:9",
              "p30:10","p35:18","p40:22","p42:22","p49:30")
$controls = @("p15","p16","p32")

"=== {0} ladder launch" -f (Get-Date -Format s) |
    Out-File -Append -Encoding utf8 $log

$tfArgs = @("-u", "$root\tests\tooling\ladder_teacher_force.py", $wd) +
          ($targets | ForEach-Object { $_.Split(":")[0] }) +
          @("--stoke", $A,
            "--sweep", "2026-08-13-m45-c-grade-sweep.jsonl",
            "--kv", "f32", "--threads", "6")

& $py @tfArgs 2>&1 | Out-File -Append -Encoding utf8 $log
$code = $LASTEXITCODE      # captured immediately; an echo here would mask it
"=== {0} teacher-force exit={1}" -f (Get-Date -Format s), $code |
    Out-File -Append -Encoding utf8 $log
if ($code -ne 0) { exit $code }

$ldArgs = @("-u", "$root\tests\tooling\c_ladder_run.py",
            "--stoke-a", $A, "--stoke-b", $B,
            "--kv-a", "f32", "--kv-b", "f32",
            "--threads", "6",
            "--workdir", $wd,
            "--out", $out,
            "--targets") + $targets + @("--controls") + $controls

& $py @ldArgs 2>&1 | Out-File -Append -Encoding utf8 $log
$code = $LASTEXITCODE
"=== {0} ladder exit={1}" -f (Get-Date -Format s), $code |
    Out-File -Append -Encoding utf8 $log
exit $code
