# Checkpoint 07: Blocked-Inverse / Neumann Prepare

Purpose: same-shape A-producer ablation using TileOps blocked-inverse /
Neumann-style prepare-A feeding the same TileOps PR1596 replay.

| Item | Value |
| --- | --- |
| Runtime image | `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10` for TileOps replay; TL0.1.8 artifact for FlashQLA-style prepare-A |
| nvcc | `12.9` in the TileOps replay runner; TL0.1.8 prepare-A uses recorded exported artifact/toolchain |
| Torch | `2.10.0+cu129` (`torch.version.cuda=12.9`) for TileOps replay |
| TileLang | `0.1.11+cu129.git65dbc983` for TileOps replay; TL0.1.8 for the external prepare-A artifact |
| Row | `TO/TO full` in the Section 11 benchmark |
| Expected evidence | `../../evidence/ladder/results/rerun_section11_ext_vs_neumann_pr1596_tl011_gpu4.jsonl` |
| Expected latency | `0.7474 ms` at `64K/H16`; adapter checkpoint: `0.7655 ms`; older archive: `0.695237 ms` |
| Blocked-inverse source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py` |
| TileOps replay source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py` |

Rerun with the same command as checkpoint 06. The benchmark emits both
`tl018_lowering_prepare_plus_tileops_replay` and
`tileops_neumann_prepare_plus_tileops_replay` in one JSONL row:

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
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_section11_ext_vs_neumann_pr1596_tl011_gpu4.jsonl"
```
