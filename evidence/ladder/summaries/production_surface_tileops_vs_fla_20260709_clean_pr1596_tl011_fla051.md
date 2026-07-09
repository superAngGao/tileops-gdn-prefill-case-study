# Production Surface Latency: Clean PR1596 / TileLang 0.1.11 / FLA 0.5.1

This summary is generated from `production_surface_tileops_vs_fla_20260709_clean_pr1596_tl011_fla051.jsonl`.
TileOps rows use clean TileOps PR1596 merge commit `79469fc0ddae584537df03e35d935575870574f6` with `dirty=false`; FLA rows use `flash-linear-attention==0.5.1`.

Runtime dependency contract:

- Python: `3.12.13`
- Torch: `2.10.0+cu129` (`torch.version.cuda=12.9`)
- TileLang: `0.1.11+cu129.git65dbc983`
- FLA package: `0.5.1`
- GPU contract: `GPU4/H200`; detected GPU `NVIDIA H200`
- Timer: CUPTI kernel-only with L2 flush, `warmup=5`, `repeat=20`, `trials=3`.

| Shape | TileOps scoped dispatch | FLA 0.5.1 reference | Public FlashQLA TL0.1.8 anchor | Speedup vs FLA 0.5.1 | Throughput ratio vs public FlashQLA anchor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32K/H16` | `0.3990 ms` | `2.1303 ms` | `0.5440 ms` | `5.34x` | `1.36x` |
| `64K/H16` | `0.7498 ms` | `4.2416 ms` | `1.3073 ms` | `5.66x` | `1.74x` |
| `128K/H16` | `1.3404 ms` | `8.4520 ms` | `2.6055 ms` | `6.31x` | `1.94x` |
| `64K/H32` | `1.3193 ms` | `5.4120 ms` | `2.5942 ms` | `4.10x` | `1.97x` |
| `64K/H64` | `2.5086 ms` | `9.8426 ms` | `6.7233 ms` | `3.92x` | `2.68x` |

The FlashQLA column is a public-environment anchor, not a same-lowering attribution comparison.
