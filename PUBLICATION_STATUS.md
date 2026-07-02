# Publication Status

Status: **public release package prepared for review**.

This repository packages the current GDN prefill AKO case study as three layers:

1. `articles/gdn-prefill-ako-case-study.md`: publication-oriented case study.
2. `reports/gdn-prefill-ako-technical-report.md`: companion evidence report.
3. `supplement/gdn-prefill-ako-si.md` plus `evidence/`: artifact index and raw
   evidence bundle.

## Current Evidence Snapshot

- Main single-shape story: `64K/H16`, BTHD, fp16, H200, archived under
  `evidence/ladder/`.
- Production dispatch surface: five serving shapes, with TileOps/FLA and public
  FlashQLA anchors archived as JSONL.
- A-producer attribution: same-input A/replay ablation rows archived under
  `evidence/ladder/results/section11_*`.
- Correctness reference: recorded vendored FLA reference, with package identity
  caveats preserved in metadata.

## Release Caveats

- The FlashQLA comparison is public-environment context, not a same-lowering
  attribution experiment.
- The FLA baseline is described as a recorded vendored FLA reference unless
  package identity is independently verified.
- The TL0.1.8-lowering FlashQLA-style prepare row is an external-lowering
  harness row, not a native current-TileLang KKT port.
- Benchmark tables require refresh if the TileOps PR head, TileLang wheel,
  docker/runtime, GPU, timer semantics, or input artifact changes.

## License / Reuse

This package is published for reading, review, and citation. No open reuse
license is granted by default; see [`LICENSE.md`](LICENSE.md). Contact the
author before reusing substantial text, figures, benchmark data, or code outside
normal quotation/citation.
