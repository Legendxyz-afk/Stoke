# M4.5 -- the attention-share profile the ruling REQUIRES before L4 may be built.
#
# > THE PRECONDITION, quoted. sec 7.5 / L4 of the 2026-08-12 ruling: "attention
# f32 twin, PROFILED FIRST. vec_dot_f32 is a scalar sequential chain, legal to
# twin under CONTRACT v2 (canonical fold = the existing scalar order). Build it
# ONLY if a post-C profile shows attention >= 15% of decode AT CTX 1024-2048."
#
# The context-scaling run on 2026-08-17 measured a share of 17.6% -- but at
# ctx ~87, and reached ctx 1024 only by EXTRAPOLATING a slope 7x beyond its data.
# The ruling names a context range on purpose, so extrapolation does not
# discharge it. This script measures inside the range.
#
# > HOW BOTH ENGINES ARE PUT AT THE SAME DEPTH. stoke reaches high context via a
# long PROMPT (prefill is batched and cheap, so this costs seconds, where
# generating to ctx 1500 would cost many minutes). llama-bench reaches it via
# `-d/--n-depth`, which prefills N tokens and then measures generation from
# there. Setting llama's -d to stoke's exact prompt-token count puts both engines
# generating 128 tokens over the same context window.
#
# Prompt token counts, measured with stoke's own tokenizer (ctx at -n 8, minus 8):
#     p00                    23 tokens  -> avg ctx over 128 gen ~= 87
#     p40                   490 tokens  -> avg ctx ~= 554
#     ctx-profile-long      1436 tokens -> avg ctx ~= 1500   <-- inside 1024-2048
# ctx-profile-long is p40+p41+p43 concatenated. It is a PROFILING input, not a
# parity input, which is why it lives in tests/data and not in tests/data/parity
# (that directory has a MANIFEST and a fixed grid; adding to it would corrupt a
# gate corpus for a perf measurement).
#
# > WHAT IS BEING CALLED "ATTENTION". Everything whose cost scales with context:
# the QK and V dot products over the KV cache, plus softmax over ctx. bench --ops
# measured softmax at 10.6 us at ctx 1024, i.e. negligible, so the context-scaling
# term is the KV dots -- which is exactly what L4 targets. Fitting
# ms/token = intercept + slope * avg_ctx separates them without a profiler:
# the intercept is the context-independent work (weights, elementwise) and
# slope * ctx is the attention term.
#
# ASCII ONLY. A Unicode delta in a .ps1 killed an earlier run of the sibling
# script: BOM-less .ps1 is read as ANSI by Windows PowerShell 5.1, and the
# mangled second byte renders as U+201D, which PowerShell accepts as a STRING
# DELIMITER.

param(
    [int]$Cycles = 3,
    [int]$NGen   = 128,
    [int]$Threads = 8,
    [string]$Stamp = "2026-08-17-m45-attention-share"
)

$ErrorActionPreference = 'Continue'

$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-att-e.txt"
$tmpO = "$env:TEMP\stoke-att-o.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"

# name, prompt file, prompt tokens (measured, not assumed)
$POINTS = @(
    @{ name = "p00";  file = "$root\tests\data\parity\p00.txt";     ptok = 23 },
    @{ name = "p40";  file = "$root\tests\data\parity\p40.txt";     ptok = 490 },
    @{ name = "long"; file = "$root\tests\data\ctx-profile-long.txt"; ptok = 1436 }
)

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 4)
}
function StokeRate($file) {
    Start-Process -FilePath $STOKE -ArgumentList @("batch", ('"' + $Q4 + '"'), "--raw",
        "-n", "$NGen", "-f", ('"' + $file + '"'), "--seqs", "1", "--threads", "$Threads",
        "--kv", "f32", "--timing") -RedirectStandardError $tmpE `
        -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpE -Raw), '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}
function LlamaRate($depth) {
    Start-Process -FilePath $LB -ArgumentList @("-m", ('"' + $Q4 + '"'), "-p", "0",
        "-n", "$NGen", "-d", "$depth", "-r", "3", "-t", "$Threads") `
        -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpO -Raw), 'tg\d+[^|]*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

Say "START cycles=$Cycles n=$NGen threads=$Threads"
$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    $null = StokeRate $POINTS[0].file
    Say "CYCLE $c"
    $order = if ($c % 2 -eq 1) { $POINTS } else { $POINTS[($POINTS.Count - 1)..0] }
    foreach ($pt in $order) {
        $pair = if ($c -le 2) { @("S", "L") } else { @("L", "S") }
        foreach ($eng in $pair) {
            $v = if ($eng -eq "S") { StokeRate $pt.file } else { LlamaRate $pt.ptok }
            $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ptok = $pt.ptok; engine = $eng; rate = $v }
            Say ("  c{0} {1,-5} ptok={2,-5} {3}  {4}" -f $c, $pt.name, $pt.ptok, $eng,
                 $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
        }
    }
}

$lines = @()
$lines += "# M4.5 -- attention share at ctx 1024-2048, the ruling's precondition for L4"
$lines += ""
$lines += "$Cycles cycles, point-order and engine-order both rotating, $Threads threads, unpinned, n=$NGen generated per measurement. stoke reaches depth via a long prompt; llama via ``-d/--n-depth`` set to stoke's exact prompt-token count, so both generate $NGen tokens over the same context window."
$lines += ""
$lines += "| point | prompt tok | avg ctx | stoke tok/s | stoke ms/tok | llama tok/s | llama ms/tok | stoke/llama |"
$lines += "|---|---|---|---|---|---|---|---|"
$fit = @{}
foreach ($pt in $POINTS) {
    $s = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.engine -eq "S" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $l = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.engine -eq "L" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $ctx = $pt.ptok + [math]::Round($NGen / 2.0)
    $sm = if ($s) { [math]::Round(1000.0 / $s, 2) } else { $null }
    $lm = if ($l) { [math]::Round(1000.0 / $l, 2) } else { $null }
    $fit[$pt.name] = @{ ctx = $ctx; s = $sm; l = $lm }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $pt.name, $pt.ptok, $ctx, $s, $sm, $l, $lm,
               $(if ($s -and $l) { "{0:n3}x" -f ($s / $l) } else { "n/a" }))
}

# Slope from the two extreme points; the middle point is the linearity check,
# kept separate so a curved cost cannot hide inside a least-squares line.
$lo = $fit[$POINTS[0].name]; $mid = $fit[$POINTS[1].name]; $hi = $fit[$POINTS[2].name]
$lines += ""
$lines += "## The fit: ms/token = intercept + slope * avg_ctx"
$lines += ""
if ($lo.s -and $hi.s -and $lo.l -and $hi.l) {
    $sSlope = ($hi.s - $lo.s) / ($hi.ctx - $lo.ctx)
    $lSlope = ($hi.l - $lo.l) / ($hi.ctx - $lo.ctx)
    $sInt = $lo.s - $sSlope * $lo.ctx
    $lInt = $lo.l - $lSlope * $lo.ctx
    $lines += ("- **stoke**: slope **{0:n4}** ms per context-token, intercept **{1:n2}** ms" -f $sSlope, $sInt)
    $lines += ("- **llama**: slope **{0:n4}** ms per context-token, intercept **{1:n2}** ms" -f $lSlope, $lInt)
    if ($lSlope -ne 0) { $lines += ("- stoke's attention cost per context-token is **{0:n1}x** llama's" -f ($sSlope / $lSlope)) }
    $lines += ("- context-independent work differs by **{0:n2} ms/token** ({1:n1}%), which is the matvec/weight-streaming side" -f ($sInt - $lInt), (100.0 * ($sInt - $lInt) / $lInt))
    $lines += ""
    $lines += "Linearity check at the middle point (avg ctx $($mid.ctx)): predicted {0:n2} ms/token, measured $($mid.s). A prediction that misses badly means the cost is not linear in context and the slope is not a single mechanism." -f ($sInt + $sSlope * $mid.ctx)
    $lines += ""
    $lines += "## THE RULING'S GATE: attention share of decode"
    $lines += ""
    $lines += "| avg ctx | stoke ms/tok | attention ms/tok | **attention share** | >= 15%? |"
    $lines += "|---|---|---|---|---|"
    foreach ($pt in $POINTS) {
        $f = $fit[$pt.name]
        if ($f.s) {
            $att = $sSlope * $f.ctx
            $share = 100.0 * $att / $f.s
            $lines += ("| {0} | {1} | {2:n2} | **{3:n1}%** | {4} |" -f $f.ctx, $f.s, $att, $share,
                       $(if ($share -ge 15.0) { "YES" } else { "no" }))
        }
    }
    $lines += ""
    $lines += "The ruling builds L4 **only** if attention is >= 15% of decode at ctx 1024-2048. The ``long`` row sits at avg ctx $($hi.ctx), inside that range, and is the row that decides."
}
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
