# Evidence Bundle

This directory contains the archived evidence behind the GDN prefill case study.
The Markdown summaries are for human review; the JSONL files are the
machine-readable source of truth.

## Headline Pointers

| Evidence target | Files |
| --- | --- |
| Main `64K/H16` story and evidence summary | [`ladder/summaries/case_study_ladder_evidence_64k_h16.md`](ladder/summaries/case_study_ladder_evidence_64k_h16.md) |
| Historical `64K/H16` rerun under TileLang `0.1.11` | [`ladder/results/rerun_011_formal_64k_h16_historical_local.jsonl`](ladder/results/rerun_011_formal_64k_h16_historical_local.jsonl) |
| Variant inventory and claim roles | [`ladder/docs/variant_inventory.md`](ladder/docs/variant_inventory.md) |
| Same-input A-producer ablation | [`ladder/summaries/section11_a_producer_ablation_64k_h16.md`](ladder/summaries/section11_a_producer_ablation_64k_h16.md) |
| Legacy replay-side diagnostics | [`ladder/summaries/a_replay_cross_ablation_64k_h16.md`](ladder/summaries/a_replay_cross_ablation_64k_h16.md) |
| Production dispatch surface, TileOps vs FLA | [`ladder/results/production_surface_tileops_vs_fla_20260701.jsonl`](ladder/results/production_surface_tileops_vs_fla_20260701.jsonl) |
| Production dispatch surface, public FlashQLA | [`ladder/results/production_surface_flashqla_20260701.jsonl`](ladder/results/production_surface_flashqla_20260701.jsonl) |
| Production-surface correctness diagnostics | [`ladder/summaries/production_surface_correctness_metrics_20260708.md`](ladder/summaries/production_surface_correctness_metrics_20260708.md), [`ladder/results/production_surface_correctness_metrics_20260708.jsonl`](ladder/results/production_surface_correctness_metrics_20260708.jsonl) |
| Evidence harness code snapshot | [`ladder/harness/`](ladder/harness/) |
| Kernel source snapshots | [`kernel_sources/`](kernel_sources/) |
| Checkpoint rerun map | [`../checkpoints/`](../checkpoints/) |

The older `production_surface_tileops_vs_fla_20260701_tmpdir.jsonl` file is
retained only as provenance for the original collection path; the publication
pointer above uses the non-`tmpdir` artifact name.

## Reading Rules

- Do not mix controlled ladder rows, public FlashQLA anchors, component
  diagnostics, and production-wrapper rows as if they were one causal ladder.
- Use `publication_role`, `causal_ladder_eligible`, and correctness metadata in
  the JSONL rows to determine what each row is allowed to support.
- FLA references are recorded vendored references unless the row metadata
  explicitly verifies package identity.
- Rows that record `dirty=true` for a TileOps worktree are archived evidence
  rows, not clean-commit reproduction claims. The harness/adapters used to
  generate the archived rows are snapshotted in [`ladder/harness/`](ladder/harness/).
- Each checkpoint has a rerun entry in [`../checkpoints/`](../checkpoints/) and
  a kernel source snapshot in [`kernel_sources/`](kernel_sources/).
