# Checkpoint 08: Scoped Dispatch Surface

Purpose: five-shape scoped serving dispatch surface.

| Item | Value |
| --- | --- |
| Runtime dependency contract | TileLang `0.1.11`, FLA `0.5.1`, Torch `2.10.0+cu129` |
| nvcc | `12.9` (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Torch | `2.10.0+cu129` (`torch.version.cuda=12.9`) |
| TileLang | `0.1.11+cu129.git65dbc983` |
| FLA | `flash-linear-attention==0.5.1` |
| TileOps source | clean PR1596 merge commit `79469fc0ddae584537df03e35d935575870574f6`, `dirty=false` |
| Variant | `tileops_final_dispatch` |
| Expected evidence | `../../evidence/ladder/results/production_surface_tileops_vs_fla_20260709_clean_pr1596_tl011_fla051.jsonl` |
| Expected shapes | `32K/H16`, `64K/H16`, `128K/H16`, `64K/H32`, `64K/H64` |
| Dispatch wrapper source | `../../evidence/kernel_sources/tileops_pr1596/tileops/ops/gated_deltanet.py` |
| Prepare source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py` |
| Replay source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py` |
| Correctness metrics | `../../evidence/ladder/results/production_surface_correctness_metrics_20260709_clean_pr1596_tl011_fla051.jsonl` |

Rerun one shape:

```bash
# Use an environment with TileLang 0.1.11, flash-linear-attention 0.5.1,
# Torch 2.10.0+cu129, and a clean checkout of TileOps commit 79469fc.
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

Rerun the five-shape latency sweep:

```bash
# Same dependency contract as above.
cd "$TILEOPS_ROOT"
export OUT="$CASE_STUDY_ROOT/evidence/ladder/results/rerun_production_surface_tileops_vs_fla.jsonl"
: > "$OUT"
for spec in \
  "32768 16 32K_H16" \
  "65536 16 64K_H16" \
  "131072 16 128K_H16" \
  "65536 32 64K_H32" \
  "65536 64 64K_H64"
do
  set -- $spec
  T="$1"
  H="$2"
  PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
  python "$GDN_HARNESS/run_ladder.py" \
    --variant ref_fla_051 \
    --variant tileops_final_dispatch \
    --seq-len "$T" --heads "$H" --dim-k 128 --dim-v 128 --chunk-size 64 \
    --dtype fp16 --seed 20260630 --warmup 5 --repeat 20 --trials 3 \
    --gpu-contract GPU4/H200 \
    --production-root "$TILEOPS_GDN_PR1596_ROOT" \
    --output "$OUT"
done
```

Rerun correctness metrics for the five-shape surface:

```bash
# Same dependency contract as above.
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
python "$GDN_HARNESS/collect_correctness_metrics.py" \
  --production-root "$TILEOPS_GDN_PR1596_ROOT" \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_production_surface_correctness.jsonl" \
  --summary "$CASE_STUDY_ROOT/evidence/ladder/summaries/rerun_production_surface_correctness.md"
```
