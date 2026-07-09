#!/usr/bin/env python3
"""Measure public TL0.1.8 FlashQLA prepare feeding TileOps PR1596 replay.

Run this inside the FlashQLA TL0.1.8 container with PR1596 mounted first on
PYTHONPATH as ``tileops``.  This is intentionally separate from the host
current-TL migration harness: the goal is a same-process measured hybrid row
with the real public FlashQLA KKT prepare and the existing TileOps replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.autograd.profiler import DeviceType


_L2_FLUSH_CACHE: torch.Tensor | None = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    dtype = torch.float16
    return {
        "q": (torch.randn(args.batch, args.seq_len, args.heads, args.dim_k, device="cuda", dtype=dtype) * args.input_scale).contiguous(),
        "k": (torch.randn(args.batch, args.seq_len, args.heads, args.dim_k, device="cuda", dtype=dtype) * args.input_scale).contiguous(),
        "v": (torch.randn(args.batch, args.seq_len, args.heads, args.dim_v, device="cuda", dtype=dtype) * args.input_scale).contiguous(),
        "g": (-torch.rand(args.batch, args.seq_len, args.heads, device="cuda", dtype=dtype)).contiguous(),
        "beta": (torch.rand(args.batch, args.seq_len, args.heads, device="cuda", dtype=dtype) * args.beta_scale).contiguous(),
    }


def _load_or_make_inputs(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    artifact = Path(args.input_artifact) if args.input_artifact else None
    if artifact and artifact.exists():
        data = torch.load(artifact, map_location="cuda")
        tensors = {name: data[name].contiguous() for name in ("q", "k", "v", "g", "beta")}
        return tensors, {
            "source": "loaded",
            "path": str(artifact),
            "sha256": _sha256_file(artifact),
        }
    tensors = _make_inputs(args)
    return tensors, {"source": "generated_unsaved", "path": None, "sha256": None}


def _get_l2_flush_cache() -> torch.Tensor:
    global _L2_FLUSH_CACHE
    if _L2_FLUSH_CACHE is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            l2_bytes = int(256e6)
        _L2_FLUSH_CACHE = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
    return _L2_FLUSH_CACHE


def _sum_kernel_time_us(kineto_results: Any) -> float:
    total_us = 0.0
    for evt in kineto_results.events():
        if evt.device_type() == DeviceType.CUDA:
            name = evt.name()
            if "vectorized_elementwise" in name and "FillFunctor" in name:
                continue
            total_us += evt.duration_ns() / 1000.0
    return total_us


def _time_cupti(fn: Callable[[], object], warmup: int, repeat: int, trials: int) -> float:
    cache = _get_l2_flush_cache()
    for _ in range(warmup):
        cache.zero_()
        fn()
    torch.cuda.synchronize()

    trial_means: list[float] = []

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        trial_means.append(_sum_kernel_time_us(prof.profiler.kineto_results) / repeat * 1e-3)

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=trials)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=on_trace_ready,
    ) as profiler:
        for _ in range(trials):
            for _ in range(repeat):
                cache.zero_()
                fn()
            profiler.step()
            for _ in range(repeat):
                cache.zero_()
                fn()
            profiler.step()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 5e-2, rtol: float = 5e-2) -> dict[str, Any]:
    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    tol = atol + rtol * e.abs()
    flat_idx = int(diff.argmax().item())
    max_index = list(torch.unravel_index(torch.tensor(flat_idx, device=diff.device), diff.shape))
    max_index = [int(x.item()) for x in max_index]
    bad = diff > tol
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "bad_count": int(bad.sum().item()),
        "numel": int(diff.numel()),
        "bad_fraction": float(bad.float().mean().item()),
        "max_index": max_index,
        "actual_at_max": float(a.flatten()[flat_idx].item()),
        "expected_at_max": float(e.flatten()[flat_idx].item()),
        "actual_nonfinite": int((~torch.isfinite(a)).sum().item()),
        "expected_nonfinite": int((~torch.isfinite(e)).sum().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--beta-scale", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--input-artifact")
    parser.add_argument("--output", default="experiments/gated_deltanet_prefill_blog_ladder/results/section11_tl018_fq_prepare_to_tileops_replay.jsonl")
    args = parser.parse_args()

    if args.batch != 1 or args.dim_k != 128 or args.dim_v != 128:
        raise ValueError("expected B=1, DK=DV=128 for the Section 11 row")

    import flash_qla
    import tilelang
    import tileops
    from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
    from flash_qla.ops.gated_delta_rule.chunk.hopper.kkt_solve import kkt_solve
    from flash_qla.ops.utils import chunk_local_cumsum
    from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
        _prefill_partitioned_initial_state_bthd,
    )
    from tileops.kernels.gated_deltanet.gdn_prefill import fused_gdr_fwd as to_fused_gdr_fwd

    tensors, input_meta = _load_or_make_inputs(args)

    def run_fq_full():
        return chunk_gated_delta_rule(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["g"],
            tensors["beta"],
            scale=1.0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

    def run_prepare():
        g_cum = chunk_local_cumsum(tensors["g"], chunk_size=64)
        A = kkt_solve(tensors["k"], tensors["beta"], chunk_size=64)
        return A, g_cum

    def run_to_replay(A: torch.Tensor, g_cum: torch.Tensor):
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = _prefill_partitioned_initial_state_bthd(
            k=tensors["k"],
            v=tensors["v"],
            A=A,
            g=g_cum,
            beta=tensors["beta"],
            chunk_size=64,
        )
        o, _h, final_state = to_fused_gdr_fwd(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            a=A,
            g=g_cum,
            b=tensors["beta"],
            scale=1.0,
            initial_state=initial_state,
            output_final_state=True,
            output_h=False,
            output_o=True,
            cu_seqlens=cu_seqlens,
            cp_seq_map=cp_seq_map,
            raw_cu_seqlens=raw_cu_seqlens,
            chunk_size=64,
        )
        return o, final_state.to(tensors["q"].dtype)

    A_cached, g_cached = run_prepare()
    ref_o, ref_state = run_fq_full()
    got_o, got_state = run_to_replay(A_cached, g_cached)
    torch.cuda.synchronize()

    def run_hybrid_full():
        A, g_cum = run_prepare()
        return run_to_replay(A, g_cum)

    timings = {
        "public_fq_full": _time_cupti(run_fq_full, args.warmup, args.repeat, args.trials),
        "public_fq_prepare_chunk_local_cumsum_plus_kkt": _time_cupti(run_prepare, args.warmup, args.repeat, args.trials),
        "tileops_replay_on_cached_public_fq_A_g": _time_cupti(lambda: run_to_replay(A_cached, g_cached), args.warmup, args.repeat, args.trials),
        "hybrid_full_public_fq_prepare_plus_tileops_replay": _time_cupti(run_hybrid_full, args.warmup, args.repeat, args.trials),
    }

    row = {
        "row": "FQ_TL018/TO_PR1596",
        "a_producer": "public_flashqla_tl018_chunk_local_cumsum_plus_kkt_solve",
        "replay_output": "tileops_pr1596_cp_preprocess_plus_fused_gdr_fwd",
        "latency_ms": timings["hybrid_full_public_fq_prepare_plus_tileops_replay"],
        "latency_breakdown_ms": timings,
        "correctness": {
            "reference": "public FlashQLA TL0.1.8 full path in same process",
            "pass_rule": "torch.allclose(o/state, ref, atol=rtol=5e-2)",
            "o": {
                **_diff_stats(got_o, ref_o),
                "allclose": bool(torch.allclose(got_o.float(), ref_o.float(), atol=5e-2, rtol=5e-2)),
            },
            "final_state": {
                **_diff_stats(got_state, ref_state),
                "allclose": bool(torch.allclose(got_state.float(), ref_state.float(), atol=5e-2, rtol=5e-2)),
            },
        },
        "shape": {
            "B": args.batch,
            "T": args.seq_len,
            "H": args.heads,
            "DK": args.dim_k,
            "DV": args.dim_v,
            "chunk_size": 64,
            "dtype": "fp16",
            "layout": "BTHD",
        },
        "timer": {
            "kind": "cupti_kernel_only_l2_flush",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "trials": args.trials,
            "include_producers": True,
        },
        "input": input_meta,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tilelang": getattr(tilelang, "__version__", "unknown"),
            "flashqla_package_file": getattr(flash_qla, "__file__", None),
            "tileops_package_file": getattr(tileops, "__file__", None),
            "tileops_git_rev": os.environ.get("TILEOPS_GIT_REV"),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    row["correctness"]["status"] = (
        "pass"
        if row["correctness"]["o"]["allclose"] and row["correctness"]["final_state"]["allclose"]
        else "fail"
    )
    print(json.dumps(row, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
