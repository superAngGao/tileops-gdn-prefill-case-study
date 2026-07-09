# Checkpoint 04: Local BTHD Wall

Purpose: local AKO wall before CP-split replay. This checkpoint shows that
local implementation work helped but did not solve the long replay dependency.

| Item | Value |
| --- | --- |
| Runtime image | `tileops-runner:nightly-tl019-fullstack-no-tileops-ldfix` |
| nvcc | `12.9` (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Torch | `2.10.0+cu128` (`torch.version.cuda=12.8`) |
| TileLang | `0.1.9` |
| Variant | `local_bthd_wall_d09c` |
| Public label | local BTHD wall checkpoint |
| Expected evidence | `../../evidence/ladder/results/formal_64k_h16_historical_local.jsonl` |
| Expected latency in archive | `5.5566 ms` at `64K/H16` |
| Kernel source snapshot | `../../evidence/kernel_sources/historical/bthdwall-d09c8f2d/` |
| Historical commit | `d09c8f2d297d8d8cc6badaa1df139014a1d7c4de` |

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
