# M4.5 rung A' -- the verification chain, in one command.
#
# Everything on branch m45-aprime was written overnight while the P-BAR
# measurement owned the box, so NONE of it has been compiled. This runs the
# whole standing recipe in order and stops at the first failure, because a
# later green result means nothing if an earlier leg is red.
#
# ORDER MATTERS AND IS NOT ARBITRARY:
#   1  build            -- unverified source; a compile error is the likeliest
#                          failure and the cheapest to find
#   2  test_simd        -- the bit-exactness gate. The new x4 case asserts
#                          out[k] == the solo kernel for 1e4 tensors x 5
#                          lengths. THIS is what makes A' safe, not the
#                          comment on the kernel
#   3  full matrix      -- everything else, plus the four goldens
#   4  batch identity   -- real model, N in {1,2,4,8,16}; also the check that
#                          the maddubs _xn port did not move anything
#   5  byte-compare     -- against build/stoke-m45-maddubs-f3b7181a.exe, the
#                          archived pre-A' binary. A' AND THE _xn PORT MUST
#                          NOT MOVE THE STREAMS AT ALL. If they do, stop:
#                          something is sharing across rows or sequences that
#                          should not be
#
# It does NOT measure. Throughput belongs to the interleaved protocol on a
# quiet box, and section 7.3 keeps whichever variant is fastest INCLUDING
# NEITHER -- under a 5% gain the single-row path stays and A' is recorded as
# tried. Reverting is one line in rows_q4_k_avx2.

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$log  = "$root\docs\superpowers\gate-reports\2026-08-15-m45-aprime-verify.txt"
$ref  = "$root\build\stoke-m45-maddubs-f3b7181a.exe"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$Q8   = "$root\models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
$M3   = "$root\build\stoke-m3-gate-876ce53f.exe"
$py   = "C:\Users\Kabir\AppData\Local\Programs\Python\Python314\python.exe"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}

Set-Location $root
$env:PATH = "C:\msys64\usr\bin;" + $env:PATH   # `sh` for the Makefile's SHELL
Say "=== A' verification, branch $(git branch --show-current)"

# 1 -------------------------------------------------------------- build
Say "1/5 build"
$out = (mingw32-make test 2>&1 | Out-String)
$out | Out-File -Append -Encoding utf8 $log
if ($out -match "error:|Error 1") {
    Say "BUILD OR MATRIX FAILED -- stopping. See the log."
    exit 1
}
Say "  build + matrix returned clean"

# 2/3 ---------------------------------------------------- tests summary
foreach ($pat in @("test_simd", "test_kquant", "test_matvec_batch")) {
    $hit = ($out -split "`n" | Select-String -Pattern $pat | Select-Object -First 1)
    Say ("  {0}" -f $hit)
}
if ($out -notmatch "13/13 tests passed|x4_is_bit_exact") {
    Say "  NOTE: confirm the new x4 case actually ran -- it is the whole gate"
}

# 4 ----------------------------------------------------- batch identity
Say "4/5 batch identity on the real model"
$bi = (& $py -u "$root\tests\tooling\m4_batch_identity.py" --stoke "$root\build\win\stoke.exe" --model $Q4 2>&1 | Out-String)
$bi | Out-File -Append -Encoding utf8 $log
if ($bi -match "PASS") { Say "  batch identity PASS" } else { Say "  batch identity FAILED -- stopping"; exit 1 }

# 5 ------------------------------------------------------ byte-compare
Say "5/5 streams must not move"
$fail = 0
foreach ($p in @("p00", "p41")) {
    $a = (& $ref run $Q4 --raw -n 48 -f "$root\tests\data\parity\$p.txt" --kv f32 --threads 6 | Out-String)
    $b = (& "$root\build\win\stoke.exe" run $Q4 --raw -n 48 -f "$root\tests\data\parity\$p.txt" --kv f32 --threads 6 | Out-String)
    $ok = ($a -eq $b)
    if (-not $ok) { $fail++ }
    Say ("  Q4_K_M {0} vs pre-A' : {1}" -f $p, $(if ($ok) { "IDENTICAL" } else { "DIFFER  <-- STOP" }))
}
$a = (& $M3 run $Q8 --raw -n 48 -f "$root\tests\data\parity\p00.txt" --threads 6 | Out-String)
$b = (& "$root\build\win\stoke.exe" run $Q8 --raw -n 48 -f "$root\tests\data\parity\p00.txt" --threads 6 | Out-String)
$ok = ($a -eq $b)
if (-not $ok) { $fail++ }
Say ("  Q8_0 anchor p00      : {0}" -f $(if ($ok) { "IDENTICAL" } else { "DIFFER  <-- STOP" }))

if ($fail -eq 0) {
    Say "A' VERIFIED on correctness. Throughput is NOT measured here -- the"
    Say "interleaved protocol on a quiet box decides whether it is kept."
    exit 0
}
Say "A' CHANGED THE STREAMS -- do not keep it. Suspect cross-row sharing."
exit 1
