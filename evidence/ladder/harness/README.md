# Evidence Harness Snapshot

This directory snapshots the experiment-only harness code used to produce and
summarize the archived GDN prefill evidence rows in this repository.

The production implementation itself lives in TileOps. The scoped serving path
discussed by the case study entered TileOps main through
[tile-ai/TileOPs#1596](https://github.com/tile-ai/TileOPs/pull/1596), merge
commit `79469fc0ddae584537df03e35d935575870574f6`. This directory is not a
copy of the production implementation; it is the public audit copy of the
benchmark harness, adapters, and external-launcher scripts used for the
archived JSONL evidence package.

## Files

| File | Role |
| --- | --- |
| `run_ladder.py` | Runs the formal ladder variants and writes JSONL rows. |
| `variants.py` | Defines variant callables, source-root activation, metadata, and publication roles. |
| `summarize_ladder.py` | Converts JSONL rows into human-readable evidence summaries. |
| `collect_correctness_metrics.py` | Collects p99/mean/L2 correctness diagnostics for the production-surface rows. |
| `run_cross_ablation.py` | Runs replay-side A/replay ablation diagnostics. |
| `run_section11_tileops_benchmark.py` | Runs the Section 11 same-shape TileOps replay benchmark rows. |
| `run_tl018_lowering_ext_to_tileops_replay.py` | Bridges TL0.1.8-lowered FlashQLA-style producer artifacts into TileOps replay. |
| `run_tl018_fq_prepare_to_tileops_replay.py` | Supports external FlashQLA prepare-A to TileOps replay experiments. |
| `tl018_fq_prepare_launcher.cu` | External launcher used for the TL0.1.8-lowering prepare-A path. |
| `kkt_bhsc_experiment.py` | Current-TileLang KKT/BHSC experiment used for rejected-row diagnostics. |
| `collect_component_breakdown.py` | Component-breakdown helper. |

## Provenance Boundary

Several archived JSONL rows record `dirty=true` for the local TileOps or
pre-merge PR1596 worktree. That means the rows should be treated as archived
evidence under their recorded metadata, not as clean-commit reproduction claims.

This snapshot makes the evidence-generation code inspectable, but it does not
turn the archived rows into a one-command public reproduction package. A clean
main/release performance claim should be collected from the merged TileOps code
under a clean checkout and recorded as a new evidence package.
