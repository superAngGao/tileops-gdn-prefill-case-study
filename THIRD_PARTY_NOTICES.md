# Third-Party Notices

This repository is published for reading, review, and citation under the reuse
terms in [`LICENSE.md`](LICENSE.md). Those repository-level terms do not remove
or replace the original notices for third-party source snapshots included for
auditability.

## Qwen FlashQLA-Derived TileLang Source Snapshots

The following files in the PR1596 source snapshot carry Qwen/Alibaba copyright
headers and MIT license notices:

- `evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/__init__.py`
- `evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/cp_fwd.py`
- `evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py`
- `evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/prepare_h.py`
- `evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/tilelang_compat.py`

These files preserve the CP-split replay schedule lineage from Qwen's public
FlashQLA project and were adapted and modified for TileOps GatedDeltaNet
prefill integration. The source snapshots are included here to make the case
study evidence auditable; the upstream project remains the appropriate source
for FlashQLA itself.

## Flash Linear Attention Utility Snapshot

The following file carries Songlin Yang / Yu Zhang copyright headers and an MIT
license notice:

- `evidence/kernel_sources/tileops_pr1596/tileops/kernels/gated_deltanet/gdn_prefill/utils.py`

It contains small utility helpers used by the TileOps GatedDeltaNet prefill
integration source snapshot.

## MIT License Text For The Files Listed Above

```text
MIT License

Copyright (c) the respective copyright holders listed in the source files.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
