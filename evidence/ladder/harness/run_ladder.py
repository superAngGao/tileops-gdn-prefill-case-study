#!/usr/bin/env python3
"""Run one or more GDN prefill blog-ladder rows and append JSONL evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_base import bench_kernel  # noqa: E402
from variants import (  # noqa: E402
    DEFAULT_PR1596_ROOT,
    VARIANTS,
    VariantUnavailable,
    env_override_metadata,
    get_variant,
    make_callable,
    variant_metadata,
)


DEFAULT_OUTPUT = THIS_DIR / "results" / "smoke_ladder.jsonl"
SHAPE_ARGS = {
    "batch": ("--batch",),
    "seq_len": ("--seq-len", "--seq_len"),
    "heads": ("--heads",),
    "dim_k": ("--dim-k", "--dim_k"),
    "dim_v": ("--dim-v", "--dim_v"),
    "chunk_size": ("--chunk-size", "--chunk_size"),
}
TIMING_ARGS = {
    "warmup": ("--warmup",),
    "repeat": ("--repeat",),
    "trials": ("--trials",),
}
SMOKE_SHAPES = {
    "blog": {
        "batch": 1,
        "seq_len": 512,
        "heads": 16,
        "dim_k": 128,
        "dim_v": 128,
        "chunk_size": 64,
    },
    "tiny": {
        "batch": 1,
        "seq_len": 512,
        "heads": 2,
        "dim_k": 64,
        "dim_v": 64,
        "chunk_size": 64,
    },
}
SMOKE_TIMING = {"warmup": 1, "repeat": 3, "trials": 1}


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def _shape_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "B": args.batch,
        "T": args.seq_len,
        "H": args.heads,
        "DK": args.dim_k,
        "DV": args.dim_v,
        "chunk_size": args.chunk_size,
        "layout": "BTHD",
        "dtype": args.dtype,
        "scale": args.scale,
        "gpu_contract": args.gpu_contract,
    }


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, check=False,
            capture_output=True, text=True, timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_commit(root: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=root)


def _git_dirty(root: Path) -> bool | None:
    out = _run(["git", "status", "--short"], cwd=root)
    return None if out is None else bool(out.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _path_meta(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    return {
        "path": str(root),
        "exists": root.exists(),
        "commit": _git_commit(root) if root.exists() else None,
        "dirty": _git_dirty(root) if root.exists() else None,
    }


def _fla_vendor_path() -> Path:
    return REPO_ROOT / ".github/runner/vendor/flash-linear-attention"


def _fla_vendor_commit() -> str | None:
    commit_file = _fla_vendor_path() / ".git_commit.txt"
    if commit_file.exists():
        return commit_file.read_text(encoding="utf-8").strip()
    return None


def _fla_root_meta() -> dict[str, Any]:
    vendor = _fla_vendor_path()
    return {
        "path": str(vendor),
        "exists": vendor.exists(),
        "commit": None,
        "dirty": None,
        "vendor_commit_file": _fla_vendor_commit(),
        "commit_note": "vendored FLA directory is not treated as an independent git worktree",
    }


def _fla_reference_identity() -> dict[str, Any]:
    package_version = _package_version("flash-linear-attention") or _package_version("fla")
    vendor = _fla_vendor_path()
    if package_version == "0.5.1":
        version_note = (
            "The environment imports flash-linear-attention package version 0.5.1; "
            "this satisfies the requested FLA reference identity for rows that "
            "record package_version=0.5.1."
        )
    else:
        version_note = (
            "The experiment plan asks for FLA 0.5.1. This environment imports "
            "the vendored FLA op source directly when the package metadata is "
            "unavailable; formal ladder rows must verify package version or map "
            "the recorded commit to the intended release."
        )
    return {
        "variant_id": "ref_fla_051",
        "requested_version": "0.5.1",
        "package_version": package_version,
        "vendor_path": str(vendor) if vendor.exists() else None,
        "vendor_commit_file": _fla_vendor_commit(),
        "version_status": (
            "package_version_verified"
            if package_version == "0.5.1"
            else "unverified_commit_based_reference"
        ),
        "version_note": version_note,
    }


def _available_roots(production_root: Path | None) -> dict[str, Any]:
    roots = {
        "tileops_repo": _path_meta(REPO_ROOT),
        "fla_vendor": _fla_root_meta(),
    }
    if production_root is not None:
        roots["production_root"] = _path_meta(production_root)
    return roots


def _used_code_root(variant_id: str, production_root: Path | None) -> dict[str, Any]:
    if variant_id == "ref_fla_051":
        return {
            "kind": "fla_import_not_resolved",
            "reason": "actual FLA source identity is attached by variants.make_callable after import succeeds",
            "fallback_vendor": _fla_root_meta(),
        }
    if variant_id == "tileops_final_dispatch":
        return {"kind": "production_root", **(_path_meta(production_root) or {})}
    return {"kind": "tileops_repo", **(_path_meta(REPO_ROOT) or {})}


def _row_run_role(args: argparse.Namespace) -> str:
    role = getattr(args, "run_role", None)
    if role:
        return role
    return "smoke" if getattr(args, "smoke", False) else "formal"


def _publication_gate(
    args: argparse.Namespace,
    *,
    decision: str | None,
    correctness_status: str | None,
) -> dict[str, Any]:
    run_role = _row_run_role(args)
    reference_identity = _fla_reference_identity()
    reference_version_verified = (
        reference_identity.get("version_status") == "package_version_verified"
    )
    warnings = []
    if not reference_version_verified:
        warnings.append(
            "FLA 0.5.1 package version is not verified; use recorded reference source identity"
        )
    if run_role != "formal":
        return {
            "run_role": run_role,
            "publication_eligible": False,
            "publication_eligibility_reason": f"run_role={run_role}",
            "reference_version_verified": reference_version_verified,
            "publication_warnings": warnings,
        }
    if decision in (None, "unavailable", "diagnostic_only", "reject"):
        return {
            "run_role": run_role,
            "publication_eligible": False,
            "publication_eligibility_reason": f"decision={decision}",
            "reference_version_verified": reference_version_verified,
            "publication_warnings": warnings,
        }
    if correctness_status != "pass":
        return {
            "run_role": run_role,
            "publication_eligible": False,
            "publication_eligibility_reason": f"correctness={correctness_status}",
            "reference_version_verified": reference_version_verified,
            "publication_warnings": warnings,
        }
    return {
        "run_role": run_role,
        "publication_eligible": True,
        "publication_eligibility_reason": "formal run with accepted decision and passing correctness",
        "reference_version_verified": reference_version_verified,
        "publication_warnings": warnings,
    }


def _environment(
    variant_id: str,
    production_root: Path | None,
    used_code_root: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gpu_name = None
    l2_bytes = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        l2_bytes = getattr(props, "L2_cache_size", None)

    clocks = _run([
        "nvidia-smi",
        "--query-gpu=clocks.sm,clocks.mem,persistence_mode",
        "--format=csv,noheader",
    ])
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])

    env = {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_l2_bytes": l2_bytes,
        "nvidia_driver": driver.splitlines()[0] if driver else None,
        "gpu_clocks": clocks.splitlines()[0] if clocks else None,
        "docker_image": os.environ.get("DOCKER_IMAGE") or os.environ.get("IMAGE_NAME"),
        "tilelang_version": _package_version("tilelang"),
        "fla_version": _package_version("flash-linear-attention") or _package_version("fla"),
        "fla_reference": _fla_reference_identity(),
        "tileops_commit": _git_commit(REPO_ROOT),
        "tileops_dirty": _git_dirty(REPO_ROOT),
        "used_code_root": used_code_root or _used_code_root(variant_id, production_root),
        "available_roots": _available_roots(production_root),
    }
    return env


def _tensor_hash(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _generate_inputs(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness")

    dtype = _dtype(args.dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    B, T, H, DK, DV = args.batch, args.seq_len, args.heads, args.dim_k, args.dim_v
    q = torch.randn(B, T, H, DK, device="cuda", dtype=dtype, generator=generator) * 0.1
    k = torch.randn(B, T, H, DK, device="cuda", dtype=dtype, generator=generator) * 0.1
    v = torch.randn(B, T, H, DV, device="cuda", dtype=dtype, generator=generator) * 0.1
    g = -torch.rand(B, T, H, device="cuda", dtype=dtype, generator=generator)
    beta = torch.rand(B, T, H, device="cuda", dtype=dtype, generator=generator) * 0.5
    initial_state = torch.zeros(B, H, DK, DV, device="cuda", dtype=torch.float32)
    tensors = {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "g": g.contiguous(),
        "beta": beta.contiguous(),
        "initial_state": initial_state.contiguous(),
    }
    meta = {
        "seed": args.seed,
        "layout": "BTHD",
        "layout_conversion": "canonical BTHD generated; variant adapters may convert explicitly",
        "contiguity": {name: tensor.is_contiguous() for name, tensor in tensors.items()},
        "dtype": {name: str(tensor.dtype).replace("torch.", "") for name, tensor in tensors.items()},
        "initial_state_zero": bool(torch.count_nonzero(initial_state).item() == 0),
        "scale": args.scale,
        "distributions": {
            "q": "torch.randn * 0.1",
            "k": "torch.randn * 0.1",
            "v": "torch.randn * 0.1",
            "g": "-torch.rand, raw log-gate input in [-1, 0)",
            "beta": "torch.rand * 0.5, range [0, 0.5)",
            "initial_state": "zeros fp32",
        },
        "g_cum_convention": "chunk-local cumulative sum inside FLA/TileOps op",
        "beta_folding_convention": "beta passed separately; producer/downstream own folding",
    }
    return tensors, meta


def _save_artifact(tensors: dict[str, torch.Tensor], path: Path | None) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensors = {k: v.detach().cpu() for k, v in tensors.items()}
    torch.save(cpu_tensors, path)
    return str(path)


def _max_abs_rel(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def _latency_record(
    fn,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    q, k, v, g, beta = (tensors[name] for name in ("q", "k", "v", "g", "beta"))
    latency_ms = bench_kernel(
        fn,
        args=(q, k, v, g, beta),
        n_warmup=args.warmup,
        n_repeat=args.repeat,
        n_trials=args.trials,
    )
    return {
        "latency_ms": latency_ms,
        "method": "benchmarks.benchmark_base.bench_kernel",
        "timer": "CUPTI kernel-only with CUDA-event fallback",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "l2_flush_policy": "flush before every warmup/timed iteration; flush kernels excluded by profiler filter where available",
        "input_clone_pool": "enabled by bench_kernel when tensor bytes permit",
        "wrapper_scope": {
            "preprocessing": "excluded except work performed inside selected op/callable",
            "layout_conversion": "included when performed by the selected experiment adapter",
            "allocation": "excluded for generated inputs; op-internal allocations included",
            "output_final_state_conversion": "included if performed by selected callable",
        },
    }


def _auto_cp_local_chunks(num_chunks: int, num_heads: int) -> int | dict[str, str]:
    env_max_local_chunks = (
        os.environ.get("TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS")
        or os.environ.get("TILEOPS_GDN_PREFILL_CP_MAX_LOCAL_CHUNKS")
    )
    if env_max_local_chunks:
        return max(int(env_max_local_chunks), 4)
    if not torch.cuda.is_available():
        return {"status": "not_observable", "reason": "CUDA unavailable"}
    import math

    sm_count = torch.cuda.get_device_properties().multi_processor_count
    max_local_chunks = 2 ** round(math.log2(math.sqrt(num_heads * num_chunks / sm_count) * 3))
    if num_heads >= 64 and num_chunks >= 512:
        max_local_chunks = max(max_local_chunks, 256)
    return max(max_local_chunks, 4)


def _dispatch_metadata(variant_id: str, args: argparse.Namespace) -> dict[str, Any]:
    base: dict[str, Any] = {
        "layout": "BTHD",
        "dtype": args.dtype,
        "seed": args.seed,
        "env_overrides": env_override_metadata(),
    }
    if variant_id == "ref_fla_051":
        return {
            **base,
            "status": "external_reference",
            "cp_segment_count": {"status": "not_applicable", "reason": "FLA external reference"},
            "max_local_chunks": {"status": "not_applicable", "reason": "FLA external reference"},
            "block_DV": {"status": "not_applicable", "reason": "FLA external reference"},
        }
    if variant_id == "generic_a_legacy":
        return {
            **base,
            "status": "observed_from_adapter",
            "schedule_branch": "legacy_bhsd_forward",
            "num_chunks": args.seq_len // args.chunk_size,
            "cp_segment_count": {"status": "not_applicable", "reason": "legacy path has no CP split"},
            "max_local_chunks": {"status": "not_applicable", "reason": "legacy path has no CP split"},
            "block_DV": {"status": "not_applicable", "reason": "legacy path does not expose CP block_DV"},
        }
    if variant_id.startswith("local_"):
        layout = "BTHD" if variant_id == "local_bthd_wall_d09c" else "BHTD/BHSD internal"
        return {
            **base,
            "status": "historical_worktree_checkpoint",
            "schedule_branch": variant_id,
            "num_chunks": args.seq_len // args.chunk_size,
            "layout": layout,
            "cp_segment_count": {"status": "not_applicable", "reason": "pre-CP historical local-AKO path"},
            "max_local_chunks": {"status": "not_applicable", "reason": "pre-CP historical local-AKO path"},
            "block_DV": {"status": "not_applicable", "reason": "pre-CP historical local-AKO path"},
        }
    if variant_id in (
        "tileops_owned_cp_generic_a",
        "tileops_owned_cp_blocked_inverse_a",
        "tileops_final_dispatch",
    ):
        num_chunks = args.seq_len // args.chunk_size
        streams = args.batch * args.heads
        use_blocksolve_prepare = (
            args.chunk_size == 64
            and args.dim_k == 128
            and args.dim_v == 128
            and args.dtype.lower() in ("fp16", "float16")
        )
        partition_enabled = (
            os.environ.get(
                "TILEOPS_GDN_PREFILL_PARTITIONED",
                os.environ.get("TILEOPS_GDN_PREFILL_CP_SPLIT", "1"),
            )
            != "0"
        )
        use_partitioned_prefill = partition_enabled and args.batch == 1 and use_blocksolve_prepare
        max_local_chunks = _auto_cp_local_chunks(num_chunks, args.heads)
        if isinstance(max_local_chunks, int) and use_partitioned_prefill:
            import math

            cp_segment_count: int | dict[str, str] = math.ceil(num_chunks / max_local_chunks)
        else:
            cp_segment_count = {"status": "not_applicable", "reason": "partitioned prefill not selected"}

        if use_partitioned_prefill:
            block_dv: int | dict[str, str] = {
                "status": "not_observable",
                "reason": "PR1596 partitioned fused_gdr_fwd path does not expose block_DV through the public op",
            }
            schedule_branch = "bthd_partitioned_prefill"
        else:
            if args.chunk_size == 64 and streams >= 64:
                block_dv = 32
            else:
                block_dv = 16
            schedule_branch = "bthd_legacy_h_recurrence"

        return {
            **base,
            "status": "inferred_from_pr1596_dispatch_source",
            "schedule_branch": schedule_branch,
            "num_chunks": num_chunks,
            "streams": streams,
            "a_producer": (
                "generic_exact_kkt_wy_inverse"
                if variant_id == "tileops_owned_cp_generic_a"
                else "blocked_inverse_blocksolve"
            ),
            "use_blocksolve_prepare": (
                False if variant_id == "tileops_owned_cp_generic_a" else use_blocksolve_prepare
            ),
            "uses_pr1596_cp_downstream": True,
            "uses_production_dispatch_wrapper": variant_id == "tileops_final_dispatch",
            "use_partitioned_prefill": use_partitioned_prefill,
            "partition_enabled": partition_enabled,
            "cp_segment_count": cp_segment_count,
            "max_local_chunks": max_local_chunks if use_partitioned_prefill else {
                "status": "not_applicable",
                "reason": "partitioned prefill not selected",
            },
            "block_DV": block_dv,
        }
    return {
        **base,
        "status": "not_observable",
        "cp_segment_count": {"status": "not_observable", "reason": "no adapter metadata implemented"},
        "max_local_chunks": {"status": "not_observable", "reason": "no adapter metadata implemented"},
        "block_DV": {"status": "not_observable", "reason": "no adapter metadata implemented"},
    }


def _unavailable_row(
    variant_id: str,
    args: argparse.Namespace,
    reason: str,
    input_hash: str | None = None,
) -> dict[str, Any]:
    meta = variant_metadata(variant_id) if variant_id in VARIANTS else {"variant_id": variant_id}
    decision = meta.get("decision", "unavailable")
    return {
        **meta,
        **_publication_gate(args, decision=decision, correctness_status="not_run"),
        "shape": _shape_dict(args),
        "input_artifact": None,
        "input_hash": input_hash,
        "correctness": {"status": "not_run", "reason": reason},
        "latency": {"status": "not_run", "reason": reason},
        "component_breakdown": {"status": "not_collected"},
        "dispatch_metadata": _dispatch_metadata(variant_id, args),
        "environment": _environment(variant_id, args.production_root),
        "decision": decision,
    }


def run_variant(
    variant_id: str,
    tensors: dict[str, torch.Tensor],
    input_meta: dict[str, Any],
    input_hash: str,
    artifact_path: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = get_variant(variant_id)
    if not spec.runnable:
        return _unavailable_row(variant_id, args, spec.status, input_hash)

    production_root = args.production_root
    try:
        fn = make_callable(
            variant_id,
            batch=args.batch,
            seq_len=args.seq_len,
            heads=args.heads,
            dim_k=args.dim_k,
            dim_v=args.dim_v,
            chunk_size=args.chunk_size,
            dtype=_dtype(args.dtype),
            production_root=production_root,
        )
    except VariantUnavailable as exc:
        return _unavailable_row(variant_id, args, str(exc), input_hash)

    used_code_root = getattr(
        fn,
        "_tileops_ladder_used_code_root",
        _used_code_root(variant_id, production_root),
    )

    q, k, v, g, beta = (tensors[name] for name in ("q", "k", "v", "g", "beta"))

    try:
        with torch.no_grad():
            actual_o, actual_state = fn(q, k, v, g, beta)
            if variant_id == "ref_fla_051":
                ref_o, ref_state = actual_o, actual_state
                reference_path = "self"
                reference_used_code_root = used_code_root
            else:
                ref_fn = make_callable(
                    "ref_fla_051",
                    batch=args.batch,
                    seq_len=args.seq_len,
                    heads=args.heads,
                    dim_k=args.dim_k,
                    dim_v=args.dim_v,
                    chunk_size=args.chunk_size,
                    dtype=_dtype(args.dtype),
                )
                reference_used_code_root = getattr(
                    ref_fn,
                    "_tileops_ladder_used_code_root",
                    _used_code_root("ref_fla_051", production_root),
                )
                ref_o, ref_state = ref_fn(q, k, v, g, beta)
                reference_path = "fla.ops.gated_delta_rule.chunk_gated_delta_rule(output_final_state=True, scale=1.0)"
            torch.cuda.synchronize()
    except Exception as exc:
        row = _unavailable_row(variant_id, args, repr(exc), input_hash)
        row["correctness"] = {"status": "failed_to_run", "reason": repr(exc)}
        row["latency"] = {"status": "not_run", "reason": "correctness/run stage failed"}
        row["environment"] = _environment(variant_id, production_root, used_code_root)
        row.update(
            _publication_gate(
                args,
                decision=row.get("decision"),
                correctness_status="failed_to_run",
            )
        )
        return row

    o_err = _max_abs_rel(actual_o, ref_o)
    state_err = _max_abs_rel(actual_state, ref_state)
    correctness_pass = bool(
        torch.allclose(actual_o.float(), ref_o.float(), atol=args.atol, rtol=args.rtol)
        and torch.allclose(
            actual_state.float(), ref_state.float(), atol=args.atol, rtol=args.rtol
        )
    )

    abi_metadata = getattr(fn, "_tileops_ladder_abi_metadata", {"status": "not_applicable"})
    abi_collector = getattr(fn, "_tileops_ladder_collect_abi_equivalence", None)
    if abi_collector is None:
        abi_equivalence = {"status": "not_applicable"}
    else:
        try:
            with torch.no_grad():
                abi_equivalence = abi_collector(q, k, v, g, beta)
                torch.cuda.synchronize()
        except Exception as exc:
            abi_equivalence = {"status": "failed", "reason": repr(exc)}

    try:
        latency = _latency_record(fn, tensors, args)
    except Exception as exc:
        latency = {"status": "failed", "reason": repr(exc)}

    meta = variant_metadata(variant_id)
    decision = meta["decision"]
    if spec.evidence_lane in ("controlled_full_op", "conditional_full_op") and not correctness_pass:
        decision = "diagnostic_only"

    return {
        **meta,
        **_publication_gate(
            args,
            decision=decision,
            correctness_status="pass" if correctness_pass else "fail",
        ),
        "shape": _shape_dict(args),
        "input_artifact": artifact_path,
        "input_hash": input_hash,
        "input_contract": input_meta,
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "pass_rule": "torch.allclose(actual.float(), reference.float(), atol=atol, rtol=rtol)",
            "reference_variant": "ref_fla_051",
            "reference_path": reference_path,
            "reference_used_code_root": reference_used_code_root,
            "comparison_dtype": "float32",
            "tolerance": {"atol": args.atol, "rtol": args.rtol},
            "diagnostic_metrics_note": "max_abs and max_rel are reported diagnostics; pass/fail is governed by pass_rule",
            "o": o_err,
            "final_state": state_err,
        },
        "latency": latency,
        "component_breakdown": {
            "status": "not_collected",
            "required_components": [
                "chunk_local_cumsum",
                "A_producer",
                "CP_preprocess_or_corrected_starts",
                "fused_replay_output",
                "other",
            ],
        },
        "dispatch_metadata": _dispatch_metadata(variant_id, args),
        "abi_metadata": abi_metadata,
        "abi_equivalence": abi_equivalence,
        "environment": _environment(variant_id, production_root, used_code_root),
        "reference_identity": _fla_reference_identity(),
        "code": {
            "harness": str(Path(__file__).resolve()),
            "harness_commit": _git_commit(REPO_ROOT),
        },
        "decision": decision,
    }


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="Variant id; repeatable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-contract", default="GPU4/H200")
    parser.add_argument(
        "--run-role",
        choices=("smoke", "formal", "diagnostic"),
        default=None,
        help="Row-level role; default is smoke with --smoke, otherwise formal.",
    )
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PR1596_ROOT)
    parser.add_argument("--artifact", type=Path, default=None, help="Optional torch.save input artifact path.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use an explicit smoke shape and short timing defaults unless args are supplied.",
    )
    parser.add_argument(
        "--smoke-shape",
        choices=sorted(SMOKE_SHAPES),
        default="blog",
        help="Shape preset used by --smoke for unspecified shape args; default keeps H16/D128.",
    )
    args = parser.parse_args()
    supplied = set()
    argv = sys.argv[1:]
    for dest, flags in {**SHAPE_ARGS, **TIMING_ARGS}.items():
        for flag in flags:
            if any(arg == flag or arg.startswith(flag + "=") for arg in argv):
                supplied.add(dest)
    args._explicit_args = supplied
    return args


def main() -> int:
    args = parse_args()
    args.run_role = _row_run_role(args)
    if args.smoke:
        supplied = getattr(args, "_explicit_args", set())
        for field, value in SMOKE_SHAPES[args.smoke_shape].items():
            if field not in supplied:
                setattr(args, field, value)
        for field, value in SMOKE_TIMING.items():
            if field not in supplied:
                setattr(args, field, value)

    for variant_id in args.variant:
        get_variant(variant_id)

    tensors, input_meta = _generate_inputs(args)
    input_hash = _tensor_hash(tensors)
    artifact_path = _save_artifact(tensors, args.artifact)
    rows = [
        run_variant(variant_id, tensors, input_meta, input_hash, artifact_path, args)
        for variant_id in args.variant
    ]
    _append_jsonl(args.output, rows)
    print(f"wrote {len(rows)} row(s) to {args.output}")
    for row in rows:
        latency = row.get("latency", {})
        lat = latency.get("latency_ms") if isinstance(latency, dict) else None
        print(
            f"{row['variant_id']}: decision={row.get('decision')} "
            f"correctness={row.get('correctness', {}).get('status')} latency_ms={lat}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
