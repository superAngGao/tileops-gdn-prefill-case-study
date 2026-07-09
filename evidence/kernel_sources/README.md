# Kernel Source Snapshots

This directory stores source snapshots for the kernel code paths referenced by
the case-study checkpoints.

The snapshots are not a replacement for the upstream repositories. They make the
evidence package self-auditing: each checkpoint can point to a concrete source
file inside this repository, while the authoritative implementation remains in
TileOps or FlashQLA.

## Layout

| Directory | Contents | Used by |
| --- | --- | --- |
| `historical/initial-f1472392/` | Historical TileOps GDN prefill source at commit `f1472392b5d41e21c689a2a870f1d451768a082b`. | Initial correct prefill checkpoint. |
| `historical/prepare-00a60b19/` | Historical TileOps GDN prefill source at commit `00a60b19b208320c31b57162e1f067169dc765b6`. | Local prepare-specialized checkpoint. |
| `historical/htile-82707454/` | Historical TileOps GDN prefill source at commit `8270745488da1244e3dd37a493488e5d47e45563`. | Local h-tile diagnostic checkpoint. |
| `historical/bthdwall-d09c8f2d/` | Historical TileOps GDN prefill source at commit `d09c8f2d297d8d8cc6badaa1df139014a1d7c4de`. | Local BTHD wall checkpoint. |
| `tileops_pr1596/` | TileOps PR1596 merge-commit source subset from `79469fc0ddae584537df03e35d935575870574f6`. | CP split, blocked-inverse prepare, final dispatch surface. |
| `flashqla_tl018_lowered/` | Lowered TL0.1.8 FlashQLA-style KKT device kernel used by the external launcher. | FlashQLA-style prepare-A / TileOps replay ablation. |

Each historical directory includes a `SOURCE_FILES.txt` list generated when the
snapshot was created. The full TileOps checkout at the listed commit is still
the recommended runnable source; these snapshots preserve the exact kernel files
needed for audit.
