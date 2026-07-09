# TileLang 0.1.11 Historical Local Rerun: GDN Prefill 64K/H16

Source JSONL:
`evidence/ladder/results/rerun_011_formal_64k_h16_historical_local.jsonl`

Purpose: current public rerun of the historical local checkpoints under the
unified TileOpsGov Docker contract.

Runtime:

| Field | Value |
| --- | --- |
| Runtime image | `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10` |
| nvcc | `12.9` (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Torch | `2.10.0+cu129` (`torch.version.cuda=12.9`) |
| TileLang | `0.1.11+cu129.git65dbc983` |
| GPU | H200 |
| Shape | `B=1,T=65536,H=16,DK=DV=128,chunk64,fp16,BTHD` |
| Timer | CUPTI kernel-only with CUDA-event fallback; warmup `10`, repeat `50`, trials `3` |
| Input hash | `sha256:a8987a2c6d16c658a1cb8ed95e409d973a3f736e2019d8719b143f18b4741513` |

The historical source roots retain the checkpoint code and include a TileLang
`0.1.11` lowering-compatibility fix for the scalar `g_last` / `g_last_val`
recurrence values.

| Registry key | Public label | Correctness | Latency ms | Use |
| --- | --- | --- | ---: | --- |
| `local_initial_prefill_f147` | initial correct prefill checkpoint | pass | `5.5318` | first measurable serving prefill op |
| `local_prepare_specialized_00a60` | local prepare-specialized checkpoint | pass | `5.3652` | local AKO positive full-op node |
| `local_h_tile_tuned_827` | local h-tile diagnostic | fail | `5.0852` | diagnostic only |
| `local_bthd_wall_d09c` | local BTHD wall checkpoint | pass | `2.9267` | Level 2 local wall row |

The older TileLang `0.1.9` summary remains available at
`formal_64k_h16_historical_local.md` for dated runtime-lineage auditing. The
case-study narrative uses the current `0.1.11` rerun values above.
