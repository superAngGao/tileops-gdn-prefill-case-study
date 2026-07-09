"""Gated DeltaNet inference prefill kernel.

The prefill path shares the chunkwise state/output kernels with training
forward, but uses an inference-only prepare kernel that produces only ``w`` and
``u``. Training-only ``Aw``/``Au`` matrices are kept in shared memory and are
not written to global memory.
"""

import functools
import math
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .gated_deltanet_fwd import (
    _LOG2E,
    _chunk_local_cumsum,
    _h_recurrence_tl,
    _output_o_tl,
)

__all__ = ["GatedDeltaNetPrefillFwdKernel"]


@functools.lru_cache(maxsize=32)
def _prefill_prepare_w_u_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Compute Gated DeltaNet prefill ``w`` and ``u`` without global Aw/Au."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        @T.prim_func
        def prefill_prepare_w_u(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_shared = T.alloc_shared([block_C, dim_v], dtype)
                g_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_shared = T.alloc_shared([block_C, dim_v], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                T.copy(
                    k[bid, hid, by * block_C : (by + 1) * block_C, :],
                    k_shared,
                    disable_tma=True,
                )
                T.copy(
                    v[bid, hid, by * block_C : (by + 1) * block_C, :],
                    v_shared,
                    disable_tma=True,
                )
                T.copy(
                    g[bid, hid, by * block_C : (by + 1) * block_C],
                    g_shared,
                    disable_tma=True,
                )
                T.copy(
                    beta[bid, hid, by * block_C : (by + 1) * block_C],
                    beta_shared,
                    disable_tma=True,
                )

                T.clear(gram_frag)
                T.gemm(k_shared, k_shared, gram_frag, transpose_B=True)

                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -gram_frag[i, j]
                        * beta_shared[i]
                        * T.exp2((g_shared[i] - g_shared[j]) * _LOG2E),
                        T.float32(0.0),
                    )
                for i, j in T.Parallel(block_C, block_C):
                    S_shared[i, j] = T.if_then_else(
                        i == j, T.float32(1.0), T.float32(0.0)
                    )

                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                for i_s, i_k in T.Parallel(block_C, dim_k):
                    k_beta_shared[i_s, i_k] = k_shared[i_s, i_k] * beta_shared[i_s]
                T.clear(w_frag)
                T.gemm(S_shared, k_beta_shared, w_frag)
                T.copy(
                    w_frag,
                    w[bid, hid, by * block_C : (by + 1) * block_C, :],
                    disable_tma=True,
                )

                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_shared[i, j] = v_shared[i, j] * beta_shared[i]
                T.clear(u_frag)
                T.gemm(S_shared, v_beta_shared, u_frag)
                T.copy(
                    u_frag,
                    u[bid, hid, by * block_C : (by + 1) * block_C, :],
                    disable_tma=True,
                )

        return prefill_prepare_w_u

    return _fused_func


@torch.library.custom_op("tileops::gated_deltanet_prefill_fwd_kernel", mutates_args=())
def _gated_deltanet_prefill_wrapped_kernel(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    h_num_stages: int,
    h_threads: int,
    h_block_v: int,
    o_threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)
    prepare_fn = _prefill_prepare_w_u_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
    )(fused_num_stages, fused_threads)
    h_fn = _h_recurrence_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_v=h_block_v,
    )(h_num_stages, h_threads)
    o_fn = _output_o_tl(batch, head, seq_len, chunk_size, dim_k, dim_v, dtype)(
        o_threads
    )
    S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    w, u = prepare_fn(k, v, g_cum, beta)
    states, v_new = h_fn(k, g_cum, w, u, S_0)
    o = o_fn(q, k, g_cum, states, v_new)
    final_state = states[:, :, -1, :, :].contiguous()
    return o, final_state


@_gated_deltanet_prefill_wrapped_kernel.register_fake
def _gated_deltanet_prefill_wrapped_kernel_fake(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    h_num_stages: int,
    h_threads: int,
    h_block_v: int,
    o_threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del dtype, fused_num_stages, fused_threads, h_num_stages, h_threads, h_block_v, o_threads
    del k, v, g, beta
    o = torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)
    final_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, final_state


class GatedDeltaNetPrefillFwdKernel(Kernel):
    """Gated DeltaNet zero-state prefill."""

    supported_archs: list[int] = [80, 89, 90]

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
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        h_block_v = 32 if self.chunk_size >= 64 else 0
        return {
            "fused_num_stages": 2,
            "fused_threads": 256,
            "h_num_stages": 2,
            "h_threads": 256,
            "h_block_v": h_block_v,
            "o_threads": 256,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gated_deltanet_prefill_wrapped_kernel(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self.config["fused_num_stages"],
            self.config["fused_threads"],
            self.config["h_num_stages"],
            self.config["h_threads"],
            self.config.get("h_block_v", 0),
            self.config["o_threads"],
            q,
            k,
            v,
            g,
            beta,
        )
