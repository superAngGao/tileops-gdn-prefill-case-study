# TileOps GDN Prefill Case Study

This repository is the public release package for the TileOps Gated DeltaNet
prefill case study. It separates the readable article, the companion technical
report, and the raw evidence bundle so readers can choose the level of detail
they need.

## Start Here

1. **Official case study:**
   [`articles/gdn-prefill-ako-case-study.md`](articles/gdn-prefill-ako-case-study.md)

   A publication-oriented technical blog. It explains the problem, the main
   mechanism, the benchmark surface, and the reusable lessons for agentic kernel
   optimization.

2. **Companion technical report:**
   [`reports/gdn-prefill-ako-technical-report.md`](reports/gdn-prefill-ako-technical-report.md)

   A readable audit trail. It keeps the controlled ladder rows, adapter
   evidence, same-input ablations, rejected candidates, ABI caveats, and
   attribution boundaries visible for expert review.

3. **Supporting information / artifact index:**
   [`supplement/gdn-prefill-ako-si.md`](supplement/gdn-prefill-ako-si.md)

   The raw-evidence guide. It points to archived JSONL files, reproduction
   commands, lowering caveats, source caveats, and supplementary diagnostics.

## Repository Map

| Path | Role |
| --- | --- |
| `articles/` | Publication-oriented case-study entry point. |
| `reports/` | Companion technical report and readable evidence narrative. |
| `supplement/` | Supporting information, artifact index, and claim guardrails. |
| `evidence/` | Archived summaries, JSONL benchmark outputs, and variant inventory. |
| `algorithm/` | Supplemental algorithm notes. |
| `archive/` | Historical drafts and planning documents retained for provenance. |

## For Reviewers

1. Read the publication-oriented story:
   [`articles/gdn-prefill-ako-case-study.md`](articles/gdn-prefill-ako-case-study.md).
2. Check the headline production-surface evidence:
   [`production_surface_tileops_vs_fla_20260701.jsonl`](evidence/ladder/results/production_surface_tileops_vs_fla_20260701.jsonl)
   and
   [`production_surface_flashqla_20260701.jsonl`](evidence/ladder/results/production_surface_flashqla_20260701.jsonl).
3. Check the prepare-A / replay attribution:
   [`section11_a_producer_ablation_64k_h16.md`](evidence/ladder/summaries/section11_a_producer_ablation_64k_h16.md).
4. Check variant roles and rejected rows:
   [`variant_inventory.md`](evidence/ladder/docs/variant_inventory.md).
5. Check rerun commands and caveats:
   [`gdn-prefill-ako-si.md`](supplement/gdn-prefill-ako-si.md).

## Evidence Boundaries

- TileOps vs FlashQLA numbers are public-environment comparisons, not
  same-lowering replay-attribution experiments.
- FLA rows are recorded vendored FLA references unless package identity is
  explicitly verified in the evidence metadata.
- FlashQLA supplied the CP-split replay schedule family. TileOps adapted that
  schedule, combined it with a blocked-inverse / Neumann-style A producer, and
  turned the result into a shape-aware production dispatch surface merged via
  [tile-ai/TileOps#1596](https://github.com/tile-ai/TileOPs/pull/1596).
- JSONL files under `evidence/` are the machine-readable source of truth for
  the archived benchmark rows.

See [`PUBLICATION_STATUS.md`](PUBLICATION_STATUS.md) for the current release
status and known caveats.

## License / Reuse

This repository is public for reading, review, and citation. No open reuse
license is granted by default; see [`LICENSE.md`](LICENSE.md).
