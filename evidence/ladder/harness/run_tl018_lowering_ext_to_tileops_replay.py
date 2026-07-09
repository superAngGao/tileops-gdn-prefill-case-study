#!/usr/bin/env python3
"""Measure TL0.1.8 lowered FlashQLA KKT feeding existing TileOps replay.

This harness keeps the replay code untouched.  It compiles the public TL0.1.8
lowered KKT device kernel plus a small PyTorch launcher, uses the current
FlashQLA/TL chunk-local cumsum for g, and feeds both tensors into the PR1596
TileOps replay path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.autograd.profiler import DeviceType
from torch.utils.cpp_extension import load


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_PR1596_ROOT = Path("/home/ga/TileOPs-pr1596")
DEFAULT_FLASHQLA_MIGRATION_SRC = Path(
    "/home/ga/Documents/gdn_kernel_bench_2026-06-18/FlashQLA-tl019-migration-src"
)
DEFAULT_INPUT_ARTIFACT = Path(
    "/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/"
    "flashqla_cross_ablation/artifacts/fq_tl018_64k_h16_seed20260630.pt"
)
DEFAULT_TL018_HEADERS = Path(
    "/home/ga/Documents/gdn_kernel_bench_2026-06-18/tilelang_tl018_headers"
)
DEFAULT_TL018_DEVICE_KERNEL = Path(
    "/home/ga/Documents/gdn_kernel_bench_2026-06-18/.cache/"
    "gdn-kernel-bench_flashqla-tl018/dot_tilelang/cache/"
    "47117e403c04ed7cdff7af46d2cfc488de1949b81a07ebd9d65d7a432dce29d2/"
    "device_kernel.cu"
)
DEFAULT_OUTPUT = THIS_DIR / "results" / "section11_tl018_lowering_ext_prepare_to_tileops_replay_64k_h16.jsonl"

_L2_FLUSH_CACHE: torch.Tensor | None = None


def _install_path(path: Path) -> None:
    resolved = str(path.resolve())
    sys.path = [p for p in sys.path if not p or str(Path(p).resolve()) != resolved]
    sys.path.insert(0, resolved)


def _clear_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def _git_commit(root: Path) -> str | None:
    import subprocess

    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_inputs(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    data = torch.load(path, map_location="cuda")
    tensors = {name: data[name].contiguous() for name in ("q", "k", "v", "g", "beta")}
    reference = {
        "A": data["A_fq"].contiguous(),
        "g_cum": data["g_cum_fq"].contiguous(),
        "ref_o": data["ref_o_fq"].contiguous(),
        "ref_state": data["ref_state_fq"].contiguous(),
    }
    return tensors, reference, {
        "source": "public_flashqla_tl018_artifact",
        "path": str(path),
        "sha256": _sha256_file(path),
        "keys": sorted(data.keys()),
    }


def _load_tl018_kkt_extension(args: argparse.Namespace):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    headers = Path(args.tl018_headers)
    tilelang_src = headers / "src"
    cutlass = headers / "3rdparty/cutlass/include"
    launcher = THIS_DIR / "tl018_fq_prepare_launcher.cu"
    device_kernel = Path(args.tl018_device_kernel)
    flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    return load(
        name=args.extension_name,
        sources=[str(device_kernel), str(launcher)],
        extra_include_paths=[str(tilelang_src), str(cutlass)],
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        verbose=args.verbose_build,
    )


def _load_components(args: argparse.Namespace) -> dict[str, Any]:
    flashqla_src = Path(args.flashqla_src)
    pr1596_root = Path(args.production_root)
    _install_path(flashqla_src)
    _install_path(pr1596_root)
    _clear_modules(("flash_qla", "tileops"))

    from flash_qla.ops.utils import chunk_local_cumsum
    from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
        _prefill_partitioned_initial_state_bthd,
    )
    from tileops.kernels.gated_deltanet.gdn_prefill import fused_gdr_fwd as to_fused_gdr_fwd

    return {
        "chunk_local_cumsum": chunk_local_cumsum,
        "to_cp_preprocess": _prefill_partitioned_initial_state_bthd,
        "to_fused_gdr_fwd": to_fused_gdr_fwd,
    }


def _get_l2_flush_cache() -> torch.Tensor:
    global _L2_FLUSH_CACHE
    if _L2_FLUSH_CACHE is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            l2_bytes = int(256e6)
        _L2_FLUSH_CACHE = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
    return _L2_FLUSH_CACHE


def _is_flush_event(name: str) -> bool:
    return (
        ("vectorized_elementwise" in name and "FillFunctor" in name)
        or name == "[CUDA memset]"
        or "cudaMemset" in name
    )


def _collect_cuda_events(kineto_results: Any) -> dict[str, float]:
    totals_us: dict[str, float] = defaultdict(float)
    for evt in kineto_results.events():
        if evt.device_type() != DeviceType.CUDA:
            continue
        name = evt.name()
        if _is_flush_event(name):
            continue
        totals_us[name] += evt.duration_ns() / 1000.0
    return dict(totals_us)


def _time_cupti(
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    trials: int,
) -> tuple[float, dict[str, float]]:
    cache = _get_l2_flush_cache()
    for _ in range(warmup):
        cache.zero_()
        fn()
    torch.cuda.synchronize()

    trial_ms: list[float] = []
    trial_events: list[dict[str, float]] = []
    for _ in range(trials):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(repeat):
                cache.zero_()
                fn()
        torch.cuda.synchronize()
        events = _collect_cuda_events(prof.profiler.kineto_results)
        total_us = sum(events.values())
        trial_ms.append(total_us / repeat * 1e-3)
        trial_events.append({name: us / repeat * 1e-3 for name, us in events.items()})

    median_idx = sorted(range(len(trial_ms)), key=lambda i: trial_ms[i])[len(trial_ms) // 2]
    top_events = dict(sorted(trial_events[median_idx].items(), key=lambda kv: kv[1], reverse=True)[:16])
    return trial_ms[median_idx], top_events


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


def _run_to_replay(c: dict[str, Any], t: dict[str, torch.Tensor], A: torch.Tensor, g_cum: torch.Tensor):
    initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = c["to_cp_preprocess"](
        k=t["k"],
        v=t["v"],
        A=A,
        g=g_cum,
        beta=t["beta"],
        chunk_size=64,
    )
    o, _h, final_state = c["to_fused_gdr_fwd"](
        q=t["q"],
        k=t["k"],
        v=t["v"],
        a=A,
        g=g_cum,
        b=t["beta"],
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
    return o, final_state.to(t["q"].dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-artifact", default=str(DEFAULT_INPUT_ARTIFACT))
    parser.add_argument("--flashqla-src", default=str(DEFAULT_FLASHQLA_MIGRATION_SRC))
    parser.add_argument("--production-root", default=str(DEFAULT_PR1596_ROOT))
    parser.add_argument("--tl018-headers", default=str(DEFAULT_TL018_HEADERS))
    parser.add_argument("--tl018-device-kernel", default=str(DEFAULT_TL018_DEVICE_KERNEL))
    parser.add_argument("--extension-name", default="tl018_fq_prepare_h16_ext_tl018hdr")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tensors, reference, input_meta = _load_inputs(Path(args.input_artifact))
    if tensors["q"].shape != (1, 65536, 16, 128):
        raise ValueError(f"expected B=1,T=65536,H=16,D=128, got {tuple(tensors['q'].shape)}")

    ext = _load_tl018_kkt_extension(args)
    components = _load_components(args)

    def run_cumsum():
        return components["chunk_local_cumsum"](tensors["g"], chunk_size=64)

    def run_kkt():
        return ext.kkt_solve_h16(tensors["k"], tensors["beta"])

    def run_prepare():
        g_cum = run_cumsum()
        A = run_kkt()
        return A, g_cum

    A_cached, g_cached = run_prepare()
    got_o, got_state = _run_to_replay(components, tensors, A_cached, g_cached)
    torch.cuda.synchronize()

    def run_replay_cached():
        return _run_to_replay(components, tensors, A_cached, g_cached)

    def run_full():
        A, g_cum = run_prepare()
        return _run_to_replay(components, tensors, A, g_cum)

    timing_items = [
        ("chunk_local_cumsum_current_tl", run_cumsum),
        ("tl018_lowering_external_kkt_only", run_kkt),
        ("prepare_cumsum_plus_tl018_lowering_external_kkt", run_prepare),
        ("tileops_replay_cached_A_g", run_replay_cached),
        ("full_tl018_lowering_prepare_plus_tileops_replay", run_full),
    ]
    timings: dict[str, float] = {}
    events: dict[str, dict[str, float]] = {}
    for name, fn in timing_items:
        latency_ms, top_events = _time_cupti(fn, args.warmup, args.repeat, args.trials)
        timings[name] = latency_ms
        events[name] = top_events

    row = {
        "row": "TL018_LOWERING_EXT/TO_PR1596",
        "a_producer": "public_flashqla_tl018_lowered_kkt_solve_via_external_launcher",
        "g_producer": "current_flashqla_tl_chunk_local_cumsum",
        "replay_output": "tileops_pr1596_cp_preprocess_plus_fused_gdr_fwd",
        "latency_ms": timings["full_tl018_lowering_prepare_plus_tileops_replay"],
        "latency_breakdown_ms": timings,
        "latency_event_breakdown_ms_per_call_top16": events,
        "component_sum_ms": (
            timings["prepare_cumsum_plus_tl018_lowering_external_kkt"]
            + timings["tileops_replay_cached_A_g"]
        ),
        "correctness": {
            "reference": "public FlashQLA TL0.1.8 exported artifact",
            "pass_rule": "A/g exact-vs-export; o/state torch.allclose(ref, atol=rtol=5e-2)",
            "A": {
                **_diff_stats(A_cached, reference["A"], atol=0.0, rtol=0.0),
                "allclose": bool(torch.equal(A_cached, reference["A"])),
            },
            "g_cum": {
                **_diff_stats(g_cached, reference["g_cum"], atol=0.0, rtol=0.0),
                "allclose": bool(torch.equal(g_cached, reference["g_cum"])),
            },
            "o": {
                **_diff_stats(got_o, reference["ref_o"]),
                "allclose": bool(torch.allclose(got_o.float(), reference["ref_o"].float(), atol=5e-2, rtol=5e-2)),
            },
            "final_state": {
                **_diff_stats(got_state, reference["ref_state"]),
                "allclose": bool(torch.allclose(got_state.float(), reference["ref_state"].float(), atol=5e-2, rtol=5e-2)),
            },
        },
        "shape": {
            "B": 1,
            "T": 65536,
            "H": 16,
            "DK": 128,
            "DV": 128,
            "chunk_size": 64,
            "dtype": "fp16",
            "layout": "BTHD",
        },
        "timer": {
            "kind": "cupti_kernel_only_l2_flush_direct_profiler",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "trials": args.trials,
        },
        "input": input_meta,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashqla_src": str(Path(args.flashqla_src).resolve()),
            "flashqla_commit": _git_commit(Path(args.flashqla_src)),
            "tileops_pr1596_root": str(Path(args.production_root).resolve()),
            "tileops_pr1596_commit": _git_commit(Path(args.production_root)),
            "tl018_headers": str(Path(args.tl018_headers).resolve()),
            "tl018_device_kernel": str(Path(args.tl018_device_kernel).resolve()),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    row["correctness"]["status"] = (
        "pass"
        if all(v["allclose"] for v in row["correctness"].values() if isinstance(v, dict) and "allclose" in v)
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
