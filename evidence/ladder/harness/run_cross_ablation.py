#!/usr/bin/env python3
"""Run A-producer / replay-output cross-ablation for GDN prefill.

This is an experiment-only harness. It intentionally stays outside the formal
ladder registry until the mixed FlashQLA/TileOps rows are validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.autograd.profiler import DeviceType


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from variants import _import_fla_chunk_rule  # noqa: E402

DEFAULT_PR1596_ROOT = Path("/home/ga/TileOPs-pr1596")
DEFAULT_FLASHQLA_MIGRATION_SRC = Path(
    "/home/ga/Documents/gdn_kernel_bench_2026-06-18/FlashQLA-tl019-migration-src"
)
DEFAULT_OUTPUT = THIS_DIR / "results" / "cross_ablation_smoke.jsonl"

_L2_FLUSH_CACHE: torch.Tensor | None = None


def _dtype(name: str) -> torch.dtype:
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


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

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _sha256_tensor_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
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
            "sha256": _sha256_tensor_file(artifact),
        }

    tensors = _make_inputs(args)
    if artifact:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in tensors.items()}, artifact)
        sha = _sha256_tensor_file(artifact)
        return tensors, {"source": "generated_and_saved", "path": str(artifact), "sha256": sha}
    return tensors, {"source": "generated_unsaved", "path": None, "sha256": None}


def _load_external_flashqla_artifact(path: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    artifact = Path(path)
    data = torch.load(artifact, map_location="cuda")
    tensors = {name: data[name].contiguous() for name in ("q", "k", "v", "g", "beta")}
    external = {
        "g_cum": data["g_cum_fq"].contiguous(),
        "A": data["A_fq"].contiguous(),
        "ref_o": data["ref_o_fq"].contiguous(),
        "ref_state": data["ref_state_fq"].contiguous(),
    }
    meta = {
        "source": "external_flashqla_tl018_artifact",
        "path": str(artifact),
        "sha256": _sha256_tensor_file(artifact),
        "keys": sorted(data.keys()),
    }
    return tensors, external, meta


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


def _load_components(args: argparse.Namespace) -> dict[str, Any]:
    flashqla_src = Path(args.flashqla_src)
    pr1596_root = Path(args.production_root)
    if not flashqla_src.exists():
        raise FileNotFoundError(f"FlashQLA migration source not found: {flashqla_src}")
    if not pr1596_root.exists():
        raise FileNotFoundError(f"PR1596 root not found: {pr1596_root}")

    _install_path(flashqla_src)
    _install_path(pr1596_root)
    _clear_modules(("flash_qla", "tileops"))

    from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
    from flash_qla.ops.gated_delta_rule.chunk.cp_context import intra_card_cp_preprocess as fq_cp_preprocess
    from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd import fused_gdr_fwd as fq_fused_gdr_fwd
    from flash_qla.ops.gated_delta_rule.chunk.hopper.kkt_solve import kkt_solve as fq_kkt_solve
    from flash_qla.ops.utils import chunk_local_cumsum as fq_chunk_local_cumsum
    from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
        _prefill_blocksolve_A_bthd,
        _prefill_partitioned_initial_state_bthd,
    )
    from tileops.kernels.gated_deltanet.gdn_prefill import fused_gdr_fwd as to_fused_gdr_fwd

    return {
        "fq_chunk_gated_delta_rule": chunk_gated_delta_rule,
        "fq_chunk_local_cumsum": fq_chunk_local_cumsum,
        "fq_kkt_solve": fq_kkt_solve,
        "fq_cp_preprocess": fq_cp_preprocess,
        "fq_fused_gdr_fwd": fq_fused_gdr_fwd,
        "to_blocksolve_A": _prefill_blocksolve_A_bthd,
        "to_cp_preprocess": _prefill_partitioned_initial_state_bthd,
        "to_fused_gdr_fwd": to_fused_gdr_fwd,
    }


def _run_fq_full(c: dict[str, Any], t: dict[str, torch.Tensor]):
    return c["fq_chunk_gated_delta_rule"](
        t["q"], t["k"], t["v"], t["g"], t["beta"],
        scale=1.0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )


def _run_fla_ref(t: dict[str, torch.Tensor]):
    chunk_gated_delta_rule, _source = _import_fla_chunk_rule()
    return chunk_gated_delta_rule(
        t["q"], t["k"], t["v"], t["g"], t["beta"],
        scale=1.0,
        output_final_state=True,
    )


def _run_fq_replay(c: dict[str, Any], t: dict[str, torch.Tensor], A: torch.Tensor, g_cum: torch.Tensor):
    initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = c["fq_cp_preprocess"](
        k=t["k"],
        v=t["v"],
        a=A,
        g=g_cum,
        b=t["beta"],
        raw_h0=None,
        raw_cu_seqlens=None,
    )
    o, _h, final_state = c["fq_fused_gdr_fwd"](
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
    return o, final_state


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


def _build_runner(
    row_id: str,
    c: dict[str, Any],
    t: dict[str, torch.Tensor],
    *,
    include_producers: bool,
    external_flashqla: dict[str, torch.Tensor] | None = None,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    g_cum: torch.Tensor | None = None
    A_fq: torch.Tensor | None = None
    A_to: torch.Tensor | None = None

    def cached_g() -> torch.Tensor:
        nonlocal g_cum
        if g_cum is None:
            g_cum = c["fq_chunk_local_cumsum"](t["g"], chunk_size=64)
        return g_cum

    def cached_fq_a() -> torch.Tensor:
        nonlocal A_fq
        if A_fq is None:
            A_fq = c["fq_kkt_solve"](t["k"], t["beta"], chunk_size=64)
        return A_fq

    def cached_to_a() -> torch.Tensor:
        nonlocal A_to
        if A_to is None:
            A_to = c["to_blocksolve_A"](t["k"], torch.zeros_like(t["beta"]), t["beta"], 64)
        return A_to

    def current_g():
        if not include_producers:
            return cached_g()
        return c["fq_chunk_local_cumsum"](t["g"], chunk_size=64)

    def current_fq_a():
        if not include_producers:
            return cached_fq_a()
        return c["fq_kkt_solve"](t["k"], t["beta"], chunk_size=64)

    def current_to_a():
        if not include_producers:
            return cached_to_a()
        return c["to_blocksolve_A"](t["k"], torch.zeros_like(t["beta"]), t["beta"], 64)

    if row_id == "FQ/FQ":
        if include_producers:
            return lambda: _run_fq_full(c, t)
        return lambda: _run_fq_replay(c, t, cached_fq_a(), cached_g())
    if row_id == "FQ18/TO":
        if external_flashqla is None:
            raise ValueError("FQ18/TO requires --external-flashqla-artifact")
        if include_producers:
            raise ValueError("FQ18/TO uses an exported TL0.1.8 A artifact; run it with replay-only timing")

        def run_fq18_to():
            return _run_to_replay(c, t, external_flashqla["A"], external_flashqla["g_cum"])

        return run_fq18_to
    if row_id == "FQ/TO":
        def run_fq_to():
            return _run_to_replay(c, t, current_fq_a(), current_g())

        return run_fq_to
    if row_id == "TO/FQ":
        def run_to_fq():
            return _run_fq_replay(c, t, current_to_a(), current_g())

        return run_to_fq
    if row_id == "TO/TO":
        def run_to_to():
            return _run_to_replay(c, t, current_to_a(), current_g())

        return run_to_to
    raise ValueError(f"unknown row_id {row_id}")


def _flashqla_a_producer_label(flashqla_src: str) -> str:
    src = str(Path(flashqla_src).resolve())
    if "tl018" in src:
        return "public_flashqla_tl018_kkt_solve"
    return "flashqla_current_tl_migration_kkt_solve"


def _row_meta(row_id: str, flashqla_src: str) -> dict[str, str]:
    a, replay = row_id.split("/")
    return {
        "a_producer": (
            "flashqla_tl018_exported_kkt_solve"
            if a == "FQ18"
            else _flashqla_a_producer_label(flashqla_src)
            if a == "FQ"
            else "tileops_blocksolve_A"
        ),
        "replay_output": "flashqla_cp_fused_gdr_fwd" if replay == "FQ" else "tileops_pr1596_cp_fused_gdr_fwd",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--beta-scale", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--rows", nargs="+", default=["FQ/FQ", "FQ/TO", "TO/FQ", "TO/TO"])
    parser.add_argument("--include-producers", action="store_true")
    parser.add_argument("--reference", choices=("fla", "fq", "fq18"), default="fla")
    parser.add_argument("--input-artifact")
    parser.add_argument("--external-flashqla-artifact")
    parser.add_argument("--flashqla-src", default=str(DEFAULT_FLASHQLA_MIGRATION_SRC))
    parser.add_argument("--production-root", default=str(DEFAULT_PR1596_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if args.chunk_size != 64 or args.dim_k != 128 or args.dim_v != 128:
        raise ValueError("cross-ablation currently expects chunk64 and DK=DV=128")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    components = _load_components(args)
    external_flashqla = None
    if args.external_flashqla_artifact:
        tensors, external_flashqla, input_meta = _load_external_flashqla_artifact(args.external_flashqla_artifact)
    else:
        tensors, input_meta = _load_or_make_inputs(args)
    if args.reference == "fla":
        ref_o, ref_state = _run_fla_ref(tensors)
        reference_name = "recorded vendored FLA chunk_gated_delta_rule"
    else:
        if args.reference == "fq18":
            if external_flashqla is None:
                raise ValueError("--reference fq18 requires --external-flashqla-artifact")
            ref_o, ref_state = external_flashqla["ref_o"], external_flashqla["ref_state"]
            reference_name = "exported public FlashQLA TL0.1.8 full path"
        else:
            ref_o, ref_state = _run_fq_full(components, tensors)
            reference_name = "FQ/FQ current-TL full path"
    torch.cuda.synchronize()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "shape": {
            "B": args.batch,
            "T": args.seq_len,
            "H": args.heads,
            "DK": args.dim_k,
            "DV": args.dim_v,
            "chunk_size": args.chunk_size,
            "layout": "BTHD",
            "dtype": args.dtype,
        },
        "seed": args.seed,
        "input": input_meta,
        "timer": {
            "kind": "cupti_kernel_only_l2_flush",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "trials": args.trials,
            "include_producers": args.include_producers,
            "reference": args.reference,
            "external_flashqla_artifact": args.external_flashqla_artifact,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "flashqla_src": str(Path(args.flashqla_src).resolve()),
            "flashqla_commit": _git_commit(Path(args.flashqla_src)),
            "flashqla_tl019_gemm_v1_mode": os.environ.get(
                "FLASHQLA_TL019_GEMM_V1_MODE", "default"
            ),
            "tileops_pr1596_root": str(Path(args.production_root).resolve()),
            "tileops_pr1596_commit": _git_commit(Path(args.production_root)),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    for row_id in args.rows:
        runner = _build_runner(
            row_id,
            components,
            tensors,
            include_producers=args.include_producers,
            external_flashqla=external_flashqla,
        )
        got_o, got_state = runner()
        torch.cuda.synchronize()
        latency_ms = _time_cupti(runner, args.warmup, args.repeat, args.trials)
        o_err = _diff_stats(got_o, ref_o)
        s_err = _diff_stats(got_state, ref_state)
        row = {
            **common,
            "row": row_id,
            **_row_meta(row_id, args.flashqla_src),
            "latency_ms": latency_ms,
            "correctness": {
                "reference": reference_name,
                "pass_rule": "torch.allclose(o/state, ref, atol=rtol=5e-2)",
                "o": {
                    **o_err,
                    "allclose": bool(torch.allclose(got_o.float(), ref_o.float(), atol=5e-2, rtol=5e-2)),
                },
                "final_state": {
                    **s_err,
                    "allclose": bool(torch.allclose(got_state.float(), ref_state.float(), atol=5e-2, rtol=5e-2)),
                },
            },
        }
        row["correctness"]["status"] = (
            "pass"
            if row["correctness"]["o"]["allclose"] and row["correctness"]["final_state"]["allclose"]
            else "fail"
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
