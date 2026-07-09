# Refreshed Section 11 A-Producer Rerun: PR1596 / TileLang 0.1.11

This rerun measures the TL0.1.8-lowered FlashQLA-style prepare-A row and the
TileOps blocked-inverse / Neumann prepare row in the same runner.

| Field | Value |
| --- | --- |
| Shape | `B=1,T=65536,H=16,DK=DV=128,chunk64,fp16,BTHD` |
| GPU | H200, `CUDA_VISIBLE_DEVICES=4` on the host |
| TileOps source | clean PR1596 merge checkout, host-verified commit `79469fc0ddae584537df03e35d935575870574f6` |
| FlashQLA source | TL0.1.8 lowered KKT kernel artifact; host-verified migration source commit `6ef4858b5446e05bd461d9658d877e548182dbcb` |
| Runtime | Torch `2.10.0+cu129`; TileLang `0.1.11+cu129.git65dbc983` for TileOps replay |
| Timer | `benchmarks.benchmark_base.bench_kernel`, warmup `5`, repeat `20`, trials `3` |
| Input artifact | `sha256:4ba1e0c0c92ade7cd415b04f57f7f8ab93ba4781437daa6f81ac899184053810` |

| Row | Latency | Correctness |
| --- | ---: | --- |
| TL0.1.8-lowering prepare + TileOps replay | `0.824524 ms` | pass vs public TL0.1.8 artifact |
| TileOps blocksolve prepare + TileOps replay | `0.747375 ms` | pass vs public TL0.1.8 artifact |
| TL0.1.8-lowering prepare only | `0.269292 ms` | exact `A/g` vs public TL0.1.8 artifact |
| TileOps blocksolve prepare only | `0.195602 ms` | component timing only |

Evidence JSONL:
[`rerun_section11_ext_vs_neumann_pr1596_tl011_gpu4.jsonl`](../results/rerun_section11_ext_vs_neumann_pr1596_tl011_gpu4.jsonl).

The older archived Section 11 file reported `0.815029 ms -> 0.695237 ms` under
its recorded July 1 environment. The refreshed current-row comparison for the
case study is `0.824524 ms -> 0.747375 ms`.
