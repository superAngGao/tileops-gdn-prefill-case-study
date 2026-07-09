# Checkpoint 06: FlashQLA-Style Prepare-A With TileOps Replay

Purpose: same-shape A-producer ablation using TL0.1.8-lowered
FlashQLA-style KKT prepare-A feeding TileOps PR1596 replay.

| Item | Value |
| --- | --- |
| Row | `TL018-lowering/TO full` |
| Expected evidence | `../../evidence/ladder/results/section11_tileops_benchmark_ext_lowering_vs_neumann_64k_h16.jsonl` |
| Expected latency in archive | `0.815029 ms` at `64K/H16` |
| Lowered KKT device kernel | `../../evidence/kernel_sources/flashqla_tl018_lowered/device_kernel.cu` |
| External launcher | `../../evidence/ladder/harness/tl018_fq_prepare_launcher.cu` |
| TileOps replay source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py` |

Rerun:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
python "$GDN_HARNESS/run_section11_tileops_benchmark.py" \
  --input-artifact /path/to/fq_tl018_64k_h16_seed20260630.pt \
  --production-root "$TILEOPS_GDN_PR1596_ROOT" \
  --tl018-device-kernel "$CASE_STUDY_ROOT/evidence/kernel_sources/flashqla_tl018_lowered/device_kernel.cu" \
  --tl018-headers /path/to/tilelang_tl018_headers \
  --flashqla-src /path/to/FlashQLA-tl019-migration-src \
  --warmup 5 --repeat 20 --trials 3 \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_section11_ext_vs_neumann.jsonl"
```
