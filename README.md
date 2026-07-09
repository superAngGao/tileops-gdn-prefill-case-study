# TileOps GDN Prefill Case Study

This repository packages a TileOps case study on agent-assisted GPU kernel
optimization for Gated DeltaNet prefill.

Gated DeltaNet prefill is a useful stress test: it wants long-context
parallelism, but it must still produce the same recurrent state as sequential
decode. The case study follows how TileOps turned that scheduling problem into
a scoped synthetic serving dispatch path using three ingredients:

- local agentic kernel optimization under fixed correctness and benchmark
  gates;
- the CP-split replay schedule family shown by Qwen's FlashQLA project;
- a TileOps-owned blocked-inverse / Neumann-style prepare-A producer.

The optimized path has merged into TileOps main through
[tile-ai/TileOps#1596](https://github.com/tile-ai/TileOPs/pull/1596). The
headline benchmark package in this repository was refreshed on the clean
PR1596 merge commit under the recorded dependency contract.

## What To Read

Start with the publication-oriented case study:

[`articles/gdn-prefill-ako-case-study.md`](articles/gdn-prefill-ako-case-study.md)

It explains the operator, the schedule changes, the benchmark surface, and the
reusable lesson: the publishable unit is not an agent-generated kernel, but an
auditable optimization loop.

Then use the companion report when you want the audit trail:

[`reports/gdn-prefill-ako-technical-report.md`](reports/gdn-prefill-ako-technical-report.md)

It keeps the controlled ladder rows, adapter evidence, same-input ablations,
rejected candidates, ABI caveats, and attribution boundaries visible.

Use the supporting information as the artifact index:

[`supplement/gdn-prefill-ako-si.md`](supplement/gdn-prefill-ako-si.md)

It points to JSONL files, reproduction commands, lowering caveats, source
caveats, and supplementary diagnostics.

Use the checkpoint map when you want to rerun a specific story node:

[`checkpoints/`](checkpoints/)

Each checkpoint folder names the variant or row, the expected evidence file, the
kernel source snapshot, and the rerun command.

## Headline Scope

The main article reports a five-shape H200 scoped synthetic serving-surface sweep for
synthetic BTHD fp16 inputs with `B=1`, `DK=DV=128`, and `chunk64`.

The headline comparisons have different roles:

| Row family | Role |
| --- | --- |
| TileOps scoped dispatch | The optimized TileOps GDN prefill path discussed in the case study. |
| FLA `0.5.1` reference | Correctness oracle and latency context for the headline TileOps/FLA surface. |
| Public FlashQLA TL0.1.8 anchor | Public-environment comparison and schedule reference with lighter provenance metadata than the clean TileOps/FLA rerun; not same-lowering attribution. |

The machine-readable source of truth for the headline surface is:

- [`production_surface_tileops_vs_fla_20260709_clean_pr1596_tl011_fla051.jsonl`](evidence/ladder/results/production_surface_tileops_vs_fla_20260709_clean_pr1596_tl011_fla051.jsonl)
- [`production_surface_flashqla_20260701.jsonl`](evidence/ladder/results/production_surface_flashqla_20260701.jsonl)
- [`production_surface_correctness_metrics_20260709_clean_pr1596_tl011_fla051.jsonl`](evidence/ladder/results/production_surface_correctness_metrics_20260709_clean_pr1596_tl011_fla051.jsonl)

## Repository Map

| Path | Role |
| --- | --- |
| `articles/` | Public case-study entry point. |
| `reports/` | Companion evidence report. |
| `supplement/` | Artifact index and claim guardrails. |
| `checkpoints/` | Per-checkpoint source map and rerun commands. |
| `evidence/` | Archived summaries, JSONL benchmark outputs, source snapshots, harness code, and variant inventory. |
| `algorithm/` | Supplemental algorithm notes. |
| `archive/` | Historical drafts and planning documents retained for provenance. |

## For Reviewers

If you are reviewing claims rather than reading the story:

1. Check the headline surface JSONL files listed above.
2. Check prepare-A / replay attribution:
   [`section11_a_producer_ablation_64k_h16.md`](evidence/ladder/summaries/section11_a_producer_ablation_64k_h16.md).
3. Check checkpoint-specific source and rerun commands:
   [`checkpoints/`](checkpoints/).
4. Check variant roles and rejected rows:
   [`variant_inventory.md`](evidence/ladder/docs/variant_inventory.md).
5. Check supplementary caveats:
   [`gdn-prefill-ako-si.md`](supplement/gdn-prefill-ako-si.md).

## Evidence Boundaries

- TileOps vs FlashQLA numbers are public-environment comparisons, not
  same-lowering replay-attribution experiments.
- Headline FLA rows use `flash-linear-attention==0.5.1`; older diagnostic rows
  may use recorded vendored FLA source snapshots and are labeled in metadata.
- FlashQLA supplied the CP-split replay schedule family. TileOps adapted that
  schedule, combined it with a blocked-inverse / Neumann-style A producer, and
  turned the result into a scoped dispatch surface in TileOps.
- JSONL files under `evidence/` are the machine-readable source of truth for
  the archived benchmark rows.
- Kernel source snapshots under `evidence/kernel_sources/` and harness code
  under `evidence/ladder/harness/` make each checkpoint auditable and rerunnable
  in a compatible TileOps/FlashQLA/H200 environment.

See [`PUBLICATION_STATUS.md`](PUBLICATION_STATUS.md) for the current release
status and known caveats.

## License / Reuse

This repository is public for reading, review, and citation. No open reuse
license is granted by default; see [`LICENSE.md`](LICENSE.md). Third-party
source snapshots keep their original notices; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
