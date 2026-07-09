# Checkpoint 02: Local Prepare-Specialized AKO

Purpose: local AKO checkpoint after specializing the prepare path.

| Item | Value |
| --- | --- |
| Variant | `local_prepare_specialized_00a60` |
| Public label | local prepare-specialized checkpoint |
| Expected evidence | `../../evidence/ladder/results/formal_64k_h16_historical_local.jsonl` |
| Expected latency in archive | `10.8353 ms` at `64K/H16` |
| Kernel source snapshot | `../../evidence/kernel_sources/historical/prepare-00a60b19/` |
| Historical commit | `00a60b19b208320c31b57162e1f067169dc765b6` |

Rerun:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
TILEOPS_GDN_HISTORY_ROOT_BASE="$TILEOPS_GDN_HISTORY_ROOT_BASE" \
python "$GDN_HARNESS/run_ladder.py" \
  --variant local_prepare_specialized_00a60 \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 --seed 20260630 --warmup 10 --repeat 50 --trials 3 \
  --gpu-contract GPU4/H200 \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_local_prepare_specialized.jsonl"
```
