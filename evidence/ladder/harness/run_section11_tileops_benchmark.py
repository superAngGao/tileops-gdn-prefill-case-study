#!/usr/bin/env python3
"""Benchmark Section 11 combined rows with TileOps benchmark infrastructure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_base import bench_kernel
from run_tl018_lowering_ext_to_tileops_replay import (
    DEFAULT_FLASHQLA_MIGRATION_SRC,
    DEFAULT_INPUT_ARTIFACT,
    DEFAULT_OUTPUT,
    DEFAULT_PR1596_ROOT,
    DEFAULT_TL018_DEVICE_KERNEL,
    DEFAULT_TL018_HEADERS,
    _diff_stats,
    _git_commit,
    _load_components,
    _load_inputs,
    _load_tl018_kkt_extension,
    _run_to_replay,
)


DEFAULT_OUTPUT_BENCH = (
    THIS_DIR / "results" / "section11_tileops_benchmark_ext_lowering_vs_neumann_64k_h16.jsonl"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bench(fn, tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> float:
    bench_args = tuple(tensors[name] for name in ("q", "k", "v", "g", "beta"))
    return bench_kernel(
        fn,
        args=bench_args,
        n_warmup=args.warmup,
        n_repeat=args.repeat,
        n_trials=args.trials,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-artifact", default=str(DEFAULT_INPUT_ARTIFACT))
    parser.add_argument("--flashqla-src", default=str(DEFAULT_FLASHQLA_MIGRATION_SRC))
    parser.add_argument("--production-root", default=str(DEFAULT_PR1596_ROOT))
    parser.add_argument("--tl018-headers", default=str(DEFAULT_TL018_HEADERS))
    parser.add_argument("--tl018-device-kernel", default=str(DEFAULT_TL018_DEVICE_KERNEL))
    parser.add_argument("--extension-name", default="tl018_fq_prepare_h16_ext_tl018hdr")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_BENCH))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tensors, reference, input_meta = _load_inputs(Path(args.input_artifact))
    ext = _load_tl018_kkt_extension(args)
    components = _load_components(args)

    from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
        _prefill_blocksolve_A_bthd,
    )

    def run_ext_full(q, k, v, g, beta):
        g_cum = components["chunk_local_cumsum"](g, chunk_size=64)
        A = ext.kkt_solve_h16(k, beta)
        local_tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        return _run_to_replay(components, local_tensors, A, g_cum)

    def run_neumann_full(q, k, v, g, beta):
        g_cum = components["chunk_local_cumsum"](g, chunk_size=64)
        A = _prefill_blocksolve_A_bthd(k, torch.zeros_like(beta), beta, 64)
        local_tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        return _run_to_replay(components, local_tensors, A, g_cum)

    def run_ext_prepare(q, k, v, g, beta):
        del q, v
        g_cum = components["chunk_local_cumsum"](g, chunk_size=64)
        A = ext.kkt_solve_h16(k, beta)
        return A, g_cum

    def run_neumann_prepare(q, k, v, g, beta):
        del q, v
        g_cum = components["chunk_local_cumsum"](g, chunk_size=64)
        A = _prefill_blocksolve_A_bthd(k, torch.zeros_like(beta), beta, 64)
        return A, g_cum

    A_ext, g_ext = run_ext_prepare(**tensors)
    got_ext_o, got_ext_state = _run_to_replay(components, tensors, A_ext, g_ext)
    A_neu, g_neu = run_neumann_prepare(**tensors)
    got_neu_o, got_neu_state = _run_to_replay(components, tensors, A_neu, g_neu)
    torch.cuda.synchronize()

    timings = {
        "tl018_lowering_prepare": _bench(run_ext_prepare, tensors, args),
        "tl018_lowering_prepare_plus_tileops_replay": _bench(run_ext_full, tensors, args),
        "tileops_neumann_prepare": _bench(run_neumann_prepare, tensors, args),
        "tileops_neumann_prepare_plus_tileops_replay": _bench(run_neumann_full, tensors, args),
    }

    row = {
        "row": "SECTION11_TILEOPS_BENCHMARK_EXT_LOWERING_VS_NEUMANN",
        "latency_ms": timings,
        "correctness": {
            "external_lowering_vs_public_tl018": {
                "A": {
                    **_diff_stats(A_ext, reference["A"], atol=0.0, rtol=0.0),
                    "allclose": bool(torch.equal(A_ext, reference["A"])),
                },
                "g_cum": {
                    **_diff_stats(g_ext, reference["g_cum"], atol=0.0, rtol=0.0),
                    "allclose": bool(torch.equal(g_ext, reference["g_cum"])),
                },
                "o": {
                    **_diff_stats(got_ext_o, reference["ref_o"]),
                    "allclose": bool(torch.allclose(got_ext_o.float(), reference["ref_o"].float(), atol=5e-2, rtol=5e-2)),
                },
                "final_state": {
                    **_diff_stats(got_ext_state, reference["ref_state"]),
                    "allclose": bool(torch.allclose(got_ext_state.float(), reference["ref_state"].float(), atol=5e-2, rtol=5e-2)),
                },
            },
            "neumann_vs_public_tl018": {
                "o": {
                    **_diff_stats(got_neu_o, reference["ref_o"]),
                    "allclose": bool(torch.allclose(got_neu_o.float(), reference["ref_o"].float(), atol=5e-2, rtol=5e-2)),
                },
                "final_state": {
                    **_diff_stats(got_neu_state, reference["ref_state"]),
                    "allclose": bool(torch.allclose(got_neu_state.float(), reference["ref_state"].float(), atol=5e-2, rtol=5e-2)),
                },
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
            "method": "benchmarks.benchmark_base.bench_kernel",
            "timer": "CUPTI kernel-only with CUDA-event fallback",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "trials": args.trials,
            "l2_flush_policy": "flush before every warmup/timed iteration; flush kernels excluded by profiler filter where available",
            "input_clone_pool": "bench_kernel-managed; disabled automatically for this >1GB clone-pool workload",
        },
        "input": {
            **input_meta,
            "sha256_rechecked": _sha256_file(Path(args.input_artifact)),
        },
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
        if (
            row["correctness"]["external_lowering_vs_public_tl018"]["A"]["allclose"]
            and row["correctness"]["external_lowering_vs_public_tl018"]["g_cum"]["allclose"]
            and row["correctness"]["external_lowering_vs_public_tl018"]["o"]["allclose"]
            and row["correctness"]["external_lowering_vs_public_tl018"]["final_state"]["allclose"]
            and row["correctness"]["neumann_vs_public_tl018"]["o"]["allclose"]
            and row["correctness"]["neumann_vs_public_tl018"]["final_state"]["allclose"]
        )
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
