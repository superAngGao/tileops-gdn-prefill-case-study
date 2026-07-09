"""Gated DeltaNet inference prefill kernel adapter.

The initial prefill kernel reuses the existing chunkwise training forward
implementation and exposes only the serving outputs: prompt output and final
recurrent state. A later inference-specialized kernel can remove the
backward-only intermediates from this path.
"""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

from .gated_deltanet_fwd import GatedDeltaNetFwdKernel

__all__ = ["GatedDeltaNetPrefillFwdKernel"]


class GatedDeltaNetPrefillFwdKernel(Kernel):
    """Gated DeltaNet zero-state prefill.

    Computes the same chunkwise recurrence as ``GatedDeltaNetFwdKernel`` but
    returns only ``(o, final_state)`` for inference prefill.
    """

    supported_archs: list[int] = GatedDeltaNetFwdKernel.supported_archs

    def __init__(
        self,
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self._fwd_kernel = GatedDeltaNetFwdKernel(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype=dtype,
            config=config,
            tune=tune,
        )
        self.config = self._fwd_kernel.config

    @property
    def default_config(self) -> dict:
        return self._fwd_kernel.default_config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, states, _Aw, _Au = self._fwd_kernel(q, k, v, g, beta)
        final_state = states[:, :, -1, :, :].contiguous()
        return o, final_state
