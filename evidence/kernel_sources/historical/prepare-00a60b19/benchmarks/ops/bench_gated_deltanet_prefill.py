"""Benchmark: TileOPs Gated DeltaNet inference prefill."""

from typing import Any

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.gated_deltanet import GatedDeltaNetPrefillFwdTest

_OP_NAME = "GatedDeltaNetPrefillFwdOp"


def _gdn_prefill_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    batch, heads, seq_len, dim_k = workload["q_shape"]
    _, _, v_seq_len, dim_v = workload["v_shape"]
    if v_seq_len != seq_len:
        raise ValueError("GDN prefill q_shape and v_shape must share seq_len")
    return batch, heads, seq_len, dim_k, dim_v, workload.get("chunk_size", 64)


_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), _gdn_prefill_args, tune=False)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune",
    _BENCH_PARAMS,
)
def test_gated_deltanet_prefill_fwd_bench(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetPrefillFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    inputs = test.gen_inputs()

    op = GatedDeltaNetPrefillFwdOp(
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        tune=tune,
    )
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
