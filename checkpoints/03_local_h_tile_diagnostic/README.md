# Checkpoint 03: Local H-Tile Diagnostic

Purpose: local h-tiling diagnostic. This row is intentionally a diagnostic
because archived correctness failed, but it remains rerunnable.

| Item | Value |
| --- | --- |
| Runtime image | `tileops-runner:nightly-tl019-fullstack-no-tileops-ldfix` |
| nvcc | `12.9` (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Torch | `2.10.0+cu128` (`torch.version.cuda=12.8`) |
| TileLang | `0.1.9` |
| Variant | `local_h_tile_tuned_827` |
| Public label | local h-tile diagnostic |
| Expected evidence | `../../evidence/ladder/results/formal_64k_h16_historical_local.jsonl` |
| Expected latency in archive | `10.1631 ms` at `64K/H16` |
| Expected correctness in archive | fail |
| Kernel source snapshot | `../../evidence/kernel_sources/historical/htile-82707454/` |
| Historical commit | `8270745488da1244e3dd37a493488e5d47e45563` |

Rerun:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
TILEOPS_GDN_HISTORY_ROOT_BASE="$TILEOPS_GDN_HISTORY_ROOT_BASE" \
python "$GDN_HARNESS/run_ladder.py" \
  --variant local_h_tile_tuned_827 \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 --seed 20260630 --warmup 10 --repeat 50 --trials 3 \
  --gpu-contract GPU4/H200 \
  --run-role diagnostic \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_local_h_tile_diagnostic.jsonl"
```
