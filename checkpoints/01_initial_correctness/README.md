# Checkpoint 01: Initial Correct Prefill

Purpose: first full-op correct TileOps GDN prefill checkpoint.

| Item | Value |
| --- | --- |
| Variant | `local_initial_prefill_f147` |
| Public label | initial correct prefill checkpoint |
| Expected evidence | `../../evidence/ladder/results/formal_64k_h16_historical_local.jsonl` |
| Expected latency in archive | `11.1762 ms` at `64K/H16` |
| Kernel source snapshot | `../../evidence/kernel_sources/historical/initial-f1472392/` |
| Historical commit | `f1472392b5d41e21c689a2a870f1d451768a082b` |

Rerun:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
TILEOPS_GDN_HISTORY_ROOT_BASE="$TILEOPS_GDN_HISTORY_ROOT_BASE" \
python "$GDN_HARNESS/run_ladder.py" \
  --variant local_initial_prefill_f147 \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 --seed 20260630 --warmup 10 --repeat 50 --trials 3 \
  --gpu-contract GPU4/H200 \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_initial_correctness.jsonl"
```
