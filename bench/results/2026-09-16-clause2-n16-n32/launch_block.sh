#!/bin/bash
# The @16 / @32 / ollama sitting, one BLOCK per invocation, launched detached from
# Git Bash the way the 09-15 sittings were (a script FILE, nohup, no harness kill):
#
#   nohup bash bench/results/2026-09-16-clause2-n16-n32/launch_block.sh 1 > /dev/null 2>&1 &
#
# Blocks (docs/superpowers/design-notes/2026-09-16-clause2-n16-n32-ollama-preregistration.md §4):
#   1  s2-closed-16  llamacpp-tuned            vs stoke-tuned@threads=12,chunk=256   3 pairs
#   2  s2-closed-32  llamacpp-tuned@ctx=22528  vs stoke-tuned@threads=12,chunk=256   3 pairs  (needs MAX_CONNS >= 32)
#   3  s2-closed-8   ollama-default            vs stoke-tuned@threads=12,chunk=256   2 pairs
#   4  s2-closed-16  ollama-default            vs stoke-tuned@threads=12,chunk=256   2 pairs
# A same-sitting llama-bench pp512 @12T referent (one-shot, no server) brackets each
# block and names the box state. Every unit is banked as it completes; re-running a
# block resumes it (--append). Verdict lines come from the harness's own output.
set -u
cd "/m/New Drive/Desktop/Programming/stoke" 2>/dev/null || cd "M:/New Drive/Desktop/Programming/stoke" || { echo "NO REPO"; exit 2; }
block="${1:?block number 1-4}"
out="bench/results/2026-09-16-clause2-n16-n32"
stoke="stoke-tuned@threads=12,chunk=256"
case "$block" in
  1) scen=s2-closed-16; armA=llamacpp-tuned;           pairs=3 ;;
  2) scen=s2-closed-32; armA=llamacpp-tuned@ctx=22528; pairs=3 ;;
  3) scen=s2-closed-8;  armA=ollama-default;           pairs=2 ;;
  4) scen=s2-closed-16; armA=ollama-default;           pairs=2 ;;
  *) echo "unknown block $block"; exit 2 ;;
esac
log="$out/block$block.log"
bank="$out/block$block.jsonl"
LB="bench/vendor/llama-b10037/llama-bench.exe"
MODEL="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
{
  echo "=== block $block: $scen  $armA vs $stoke  pairs=$pairs  start $(date +%FT%T) ==="
  echo "=== binary: $(sha256sum build/win/stoke.exe | cut -c1-16)  HEAD $(git rev-parse --short HEAD)  dirty=$(git status --short | wc -l) ==="
  echo "=== referent BEFORE: $("$LB" -m "$MODEL" -p 512 -n 0 -t 12 2>/dev/null | grep pp512) ==="
  python tests/tooling/run_e2e_interleaved.py --scenario "$scen" --arm-a "$armA" --arm-b "$stoke" \
      --pairs "$pairs" --cooldown 45 --timeout-s 900 --out "$bank" --append
  echo "=== harness rc=$? ==="
  echo "=== referent AFTER: $("$LB" -m "$MODEL" -p 512 -n 0 -t 12 2>/dev/null | grep pp512) ==="
  echo "=== block $block END $(date +%FT%T) ==="
} >> "$log" 2>&1
