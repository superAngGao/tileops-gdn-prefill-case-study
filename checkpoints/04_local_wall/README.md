# Checkpoint 04: Local BTHD Wall

Purpose: local AKO wall before CP-split replay. This checkpoint shows that
local implementation work helped but did not solve the long replay dependency.

| Item | Value |
| --- | --- |
| Runtime image | `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10` |
| nvcc | `12.9` (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Torch | `2.10.0+cu129` (`torch.version.cuda=12.9`) |
| TileLang | `0.1.11+cu129.git65dbc983` |
| Variant | `local_bthd_wall_d09c` |
| Public label | local BTHD wall checkpoint |
| Current 0.1.11 rerun evidence | `../../evidence/ladder/results/rerun_011_formal_64k_h16_historical_local.jsonl` |
| Current 0.1.11 rerun latency | `2.9267 ms` at `64K/H16` |
| Historical archive evidence | `../../evidence/ladder/results/formal_64k_h16_historical_local.jsonl` |
| Historical archive latency | `5.5566 ms` at `64K/H16` |
| Kernel source snapshot | `../../evidence/kernel_sources/historical/bthdwall-d09c8f2d/` |
| Historical commit | `d09c8f2d297d8d8cc6badaa1df139014a1d7c4de` |
| Rerun source note | Historical checkpoint source with a TileLang `0.1.11` scalar-lowering compatibility fix for `g_last` / `g_last_val`. |

Rerun:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
TILEOPS_GDN_HISTORY_ROOT_BASE="$TILEOPS_GDN_HISTORY_ROOT_BASE" \
python "$GDN_HARNESS/run_ladder.py" \
  --variant local_bthd_wall_d09c \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 --seed 20260630 --warmup 10 --repeat 50 --trials 3 \
  --gpu-contract GPU4/H200 \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_local_wall.jsonl"
```
