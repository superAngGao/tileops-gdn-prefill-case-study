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

## Evidence Boundaries

- TileOps vs FlashQLA numbers are public-environment comparisons, not
  same-lowering replay-attribution experiments.
- FLA rows are recorded vendored FLA references unless package identity is
  explicitly verified in the evidence metadata.
- FlashQLA supplied the CP-split replay schedule family. TileOps adapted that
  schedule, combined it with a blocked-inverse / Neumann-style A producer, and
  turned the result into a shape-aware production dispatch surface.
- JSONL files under `evidence/` are the machine-readable source of truth for
  the archived benchmark rows.

See [`PUBLICATION_STATUS.md`](PUBLICATION_STATUS.md) for the current release
status and known caveats.
