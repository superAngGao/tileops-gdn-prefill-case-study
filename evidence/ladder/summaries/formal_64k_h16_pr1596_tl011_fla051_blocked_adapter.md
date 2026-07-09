# Formal 64K/H16 PR1596 TileLang 0.1.11 Blocked-Adapter Rerun

This rerun uses the clean PR1596 merge commit with TileLang `0.1.11`, Torch `2.10.0+cu129`, and FLA `0.5.1`. It checks the explicit blocked-inverse / Neumann adapter against the dispatch wrapper under the same dependency contract used by the refreshed production surface.

| Row | Latency | Correctness | Source |
| --- | ---: | --- | --- |
| `ref_fla_051` | `4.243957 ms` | pass | commit `n/a`, dirty `None` |
| `tileops_owned_cp_blocked_inverse_a` | `0.765505 ms` | pass | commit `79469fc0ddae584537df03e35d935575870574f6`, dirty `False` |
| `tileops_final_dispatch` | `0.748071 ms` | pass | commit `79469fc0ddae584537df03e35d935575870574f6`, dirty `False` |

Evidence JSONL: [`formal_64k_h16_pr1596_tl011_fla051_blocked_adapter.jsonl`](../results/formal_64k_h16_pr1596_tl011_fla051_blocked_adapter.jsonl).

Use `tileops_owned_cp_blocked_inverse_a` as the clean TileLang `0.1.11` Neumann adapter checkpoint. Older Section 11 `0.695237 ms` rows remain archived A-producer ablation evidence and should not be used as the clean checkpoint latency.
