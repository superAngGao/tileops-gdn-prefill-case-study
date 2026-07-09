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
- Scoped synthetic dispatch surface: five serving shapes, with TileOps/FLA and public
  FlashQLA anchors archived as JSONL. The TileOps GDN prefill path entered
  TileOps main through
  [tile-ai/TileOps#1596](https://github.com/tile-ai/TileOPs/pull/1596), merge
  commit `79469fc0ddae584537df03e35d935575870574f6`.
- A-producer attribution: same-input A/replay ablation rows archived under
  `evidence/ladder/results/section11_*`.
- Correctness reference: `flash-linear-attention==0.5.1` for the headline
  clean PR1596 surface; older diagnostics may use recorded vendored FLA source
  snapshots and keep that caveat in metadata. The five-shape scoped surface has an
  archived correctness-metrics refresh with p95/p99 absolute error, mean
  absolute error, L2 norm-relative error, nonfinite counts, and input hashes:
  `evidence/ladder/summaries/production_surface_correctness_metrics_20260709_clean_pr1596_tl011_fla051.md`.

## Release Caveats

- The FlashQLA comparison is public-environment context, not a same-lowering
  attribution experiment.
- Headline FLA rows use `flash-linear-attention==0.5.1`. Older diagnostics that
  rely on vendored FLA snapshots remain labeled as recorded vendored references.
- The TL0.1.8-lowering FlashQLA-style prepare row is an external-lowering
  harness row, not a native current-TileLang KKT port.
- Benchmark tables require refresh if the TileOps main/release commit,
  TileLang wheel, docker/runtime, GPU, timer semantics, or input artifact
  changes.

## License / Reuse

This package is published for reading, review, and citation. No open reuse
license is granted by default; see [`LICENSE.md`](LICENSE.md). Contact the
author before reusing substantial text, figures, benchmark data, or code outside
normal quotation/citation. Third-party source snapshots keep their original
notices; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
