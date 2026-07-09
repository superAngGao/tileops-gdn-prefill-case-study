# Checkpoint 08: Scoped Dispatch Surface

Purpose: five-shape scoped serving dispatch surface.

| Item | Value |
| --- | --- |
| Variant | `tileops_final_dispatch` |
| Expected evidence | `../../evidence/ladder/results/production_surface_tileops_vs_fla_20260701.jsonl` |
| Expected shapes | `32K/H16`, `64K/H16`, `128K/H16`, `64K/H32`, `64K/H64` |
| Dispatch wrapper source | `../../evidence/kernel_sources/tileops_pr1596/tileops/ops/gated_deltanet.py` |
| Prepare source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py` |
| Replay source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py` |
| Correctness metrics | `../../evidence/ladder/results/production_surface_correctness_metrics_20260708.jsonl` |

Rerun one shape:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
python "$GDN_HARNESS/run_ladder.py" \
  --variant ref_fla_051 \
  --variant tileops_final_dispatch \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 --seed 20260630 --warmup 5 --repeat 20 --trials 3 \
  --gpu-contract GPU4/H200 \
  --production-root "$TILEOPS_GDN_PR1596_ROOT" \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_dispatch_64k_h16.jsonl"
```

Rerun correctness metrics for the five-shape surface:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
python "$GDN_HARNESS/collect_correctness_metrics.py" \
  --production-root "$TILEOPS_GDN_PR1596_ROOT" \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_production_surface_correctness.jsonl" \
  --summary "$CASE_STUDY_ROOT/evidence/ladder/summaries/rerun_production_surface_correctness.md"
```
