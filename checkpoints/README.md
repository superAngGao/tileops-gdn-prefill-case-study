# Checkpoint Reproducibility Map

This directory maps each case-study checkpoint to the code and command needed to
rerun it in a compatible H200/TileOps/FlashQLA environment.

All checkpoints are intended to be rerunnable. The repository includes the
harness, kernel source snapshots, and compressed full source roots for the
historical TileOps checkpoints. Some rows still require external benchmark
artifacts, such as the exported FlashQLA TL0.1.8 artifact used by the
same-shape A-producer ablation.

Runtime split:

- Historical local checkpoints (`01`-`04`) were collected with the TileLang
  `0.1.9` runner lineage. The local rerun-verified image is
  `tileops-runner:nightly-tl019-fullstack-no-tileops-ldfix`.
- Current/merged TileOps rows (`05`, `07`, `08`) rerun under the TileOpsGov CI
  image `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10`.
- The FlashQLA-style prepare-A row (`06`) also needs the exported TL0.1.8
  artifact and launcher inputs recorded in its checkpoint README.

Environment pieces:

- a TileOps checkout at the listed commit;
- the evidence harness snapshot in `../evidence/ladder/harness/`;
- the FlashQLA TL0.1.8 environment or exported TL0.1.8 artifact for Section 11;
- H200/CUDA/TileLang/PyTorch runtime compatible with the archived JSONL rows.

The source snapshots under `../evidence/kernel_sources/` make the kernel code
auditable inside this repository. For historical reruns, the full source roots
can be reconstructed from:

```text
../evidence/kernel_sources/runnable_roots/*.tar.gz
```

| Checkpoint | Folder | Rerun status |
| --- | --- | --- |
| Initial correct prefill | `01_initial_correctness/` | rerunnable with historical TileOps root |
| Local prepare-specialized AKO | `02_local_prepare_specialized/` | rerunnable with historical TileOps root |
| Local h-tile diagnostic | `03_local_h_tile_diagnostic/` | rerunnable diagnostic with historical TileOps root |
| Local BTHD wall | `04_local_wall/` | rerunnable with historical TileOps root |
| CP-split bridge | `05_cp_split_bridge/` | rerunnable with TileOps PR1596 root and harness |
| FlashQLA-style prepare-A ablation | `06_flashqla_style_prepare/` | rerunnable with TL0.1.8 artifact/environment and TileOps PR1596 root |
| Blocked-inverse / Neumann prepare | `07_neumann_prepare/` | rerunnable with TileOps PR1596 root and Section 11 artifact |
| Scoped dispatch surface | `08_dispatch_surface/` | rerunnable with TileOps PR1596 or merged-main root |

Common environment variables used by the commands:

```bash
export CASE_STUDY_ROOT=/path/to/tileops-gdn-prefill-case-study
export TILEOPS_ROOT=/path/to/TileOPs
export GDN_HARNESS="$CASE_STUDY_ROOT/evidence/ladder/harness"
export TILEOPS_GDN_PR1596_ROOT=/path/to/TileOPs-pr1596-or-merged-main
export TILEOPS_GDN_HISTORY_ROOT_BASE=/path/to/TileOPs-gdn-history
```

The historical root base should contain:

```text
initial-f1472392/
prepare-00a60b19/
htile-82707454/
bthdwall-d09c8f2d/
```

Each subdirectory should be a TileOps checkout at the commit listed in the
corresponding checkpoint README. To reconstruct it from this repository:

```bash
mkdir -p "$TILEOPS_GDN_HISTORY_ROOT_BASE"
tar -C "$TILEOPS_GDN_HISTORY_ROOT_BASE" -xzf \
  "$CASE_STUDY_ROOT/evidence/kernel_sources/runnable_roots/initial-f1472392.tar.gz"
tar -C "$TILEOPS_GDN_HISTORY_ROOT_BASE" -xzf \
  "$CASE_STUDY_ROOT/evidence/kernel_sources/runnable_roots/prepare-00a60b19.tar.gz"
tar -C "$TILEOPS_GDN_HISTORY_ROOT_BASE" -xzf \
  "$CASE_STUDY_ROOT/evidence/kernel_sources/runnable_roots/htile-82707454.tar.gz"
tar -C "$TILEOPS_GDN_HISTORY_ROOT_BASE" -xzf \
  "$CASE_STUDY_ROOT/evidence/kernel_sources/runnable_roots/bthdwall-d09c8f2d.tar.gz"
```
