# Checkpoint 05: CP-Split Bridge

Purpose: first correct TileOps-owned CP downstream bridge. This is bridge
evidence, not a FlashQLA reproduction.

| Item | Value |
| --- | --- |
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
