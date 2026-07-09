# Checkpoint 05: CP-Split Bridge

Purpose: first correct TileOps-owned CP downstream bridge. This is bridge
evidence, not a FlashQLA reproduction.

| Item | Value |
| --- | --- |
| Runtime image | `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10` |
| nvcc | `12.9` (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Torch | `2.10.0+cu129` (`torch.version.cuda=12.9`) |
| TileLang | `0.1.11+cu129.git65dbc983` |
| Variant | `tileops_owned_cp_generic_a` |
| Public label | generic-A CP bridge |
| Expected evidence | `../../evidence/ladder/results/formal_64k_h16_v5_ladder.jsonl` |
| Expected latency in archive | `5.3912 ms` at `64K/H16` |
| Generic-A source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/fused_prepare_compute_w_u.py` |
| CP downstream source | `../../evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py` |

Rerun:

```bash
cd "$TILEOPS_ROOT"
PYTHONPATH="$TILEOPS_ROOT:$GDN_HARNESS:$PYTHONPATH" \
python "$GDN_HARNESS/run_ladder.py" \
  --variant tileops_owned_cp_generic_a \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 --seed 20260630 --warmup 10 --repeat 50 --trials 3 \
  --gpu-contract GPU4/H200 \
  --production-root "$TILEOPS_GDN_PR1596_ROOT" \
  --output "$CASE_STUDY_ROOT/evidence/ladder/results/rerun_cp_split_bridge.jsonl"
```
