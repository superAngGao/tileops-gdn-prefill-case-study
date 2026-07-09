#!/usr/bin/env python3
"""Collect distribution-level correctness metrics for headline GDN rows.

This script is intentionally correctness-only. It reuses the blog ladder input
contract and variant registry, compares TileOps final dispatch against the
recorded FLA reference, and writes JSONL plus a compact markdown summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_ladder import (  # noqa: E402
    _dtype,
    _environment,
    _fla_reference_identity,
    _generate_inputs,
    _git_commit,
    _git_dirty,
    _shape_dict,
    _tensor_hash,
)
from variants import DEFAULT_PR1596_ROOT, make_callable  # noqa: E402


HEADLINE_SHAPES = (
    ("32K/H16", 32768, 16),
    ("64K/H16", 65536, 16),
    ("128K/H16", 131072, 16),
    ("64K/H32", 65536, 32),
    ("64K/H64", 65536, 64),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _shape_args(args: argparse.Namespace, label: str, seq_len: int, heads: int) -> argparse.Namespace:
    return SimpleNamespace(
        batch=args.batch,
        seq_len=seq_len,
        heads=heads,
        dim_k=args.dim_k,
        dim_v=args.dim_v,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        seed=args.seed,
        scale=args.scale,
        gpu_contract=args.gpu_contract,
        production_root=args.production_root,
        label=label,
    )


def _sample_for_quantiles(
    diff: torch.Tensor,
    *,
    sample_limit: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    flat = diff.flatten()
    numel = flat.numel()
    if numel <= sample_limit:
        return flat, {
            "quantile_method": "exact",
            "sample_size": numel,
            "numel": numel,
        }

    stride = max(math.ceil(numel / sample_limit), 1)
    sample = flat[::stride]
    if sample.numel() > sample_limit:
        sample = sample[:sample_limit]
    return sample, {
        "quantile_method": "deterministic_even_stride_sample",
        "sample_stride": stride,
        "sample_size": sample.numel(),
        "numel": numel,
    }


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    sample_limit: int,
) -> dict[str, Any]:
    a = actual.detach().float()
    r = reference.detach().float()
    diff = (a - r).abs()
    sample, sampling = _sample_for_quantiles(diff, sample_limit=sample_limit)
    quantiles = torch.quantile(
        sample,
        torch.tensor([0.95, 0.99], device=sample.device, dtype=sample.dtype),
    )
    ref_norm = torch.linalg.vector_norm(r)
    diff_norm = torch.linalg.vector_norm(a - r)
    denom = torch.clamp(ref_norm, min=1e-12)
    rel_denom = r.abs().clamp_min(1e-6)

    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p95_abs": float(quantiles[0].item()),
        "p99_abs": float(quantiles[1].item()),
        "norm_rel_l2": float((diff_norm / denom).item()),
        "max_rel_diagnostic": float((diff / rel_denom).max().item()),
        "nonfinite_actual": int((~torch.isfinite(a)).sum().item()),
        "nonfinite_reference": int((~torch.isfinite(r)).sum().item()),
        "nonfinite_diff": int((~torch.isfinite(diff)).sum().item()),
        **sampling,
    }


def _collect_one(args: argparse.Namespace, label: str, seq_len: int, heads: int) -> dict[str, Any]:
    shape_args = _shape_args(args, label, seq_len, heads)
    tensors, input_meta = _generate_inputs(shape_args)
    input_hash = _tensor_hash(tensors)
    dtype = _dtype(args.dtype)

    tileops_fn = make_callable(
        "tileops_final_dispatch",
        batch=args.batch,
        seq_len=seq_len,
        heads=heads,
        dim_k=args.dim_k,
        dim_v=args.dim_v,
        chunk_size=args.chunk_size,
        dtype=dtype,
        production_root=args.production_root,
    )
    tileops_used_code_root = getattr(tileops_fn, "_tileops_ladder_used_code_root", None)

    ref_fn = make_callable(
        "ref_fla_051",
        batch=args.batch,
        seq_len=seq_len,
        heads=heads,
        dim_k=args.dim_k,
        dim_v=args.dim_v,
        chunk_size=args.chunk_size,
        dtype=dtype,
        production_root=args.production_root,
    )
    reference_used_code_root = getattr(ref_fn, "_tileops_ladder_used_code_root", None)

    q, k, v, g, beta = (tensors[name] for name in ("q", "k", "v", "g", "beta"))
    with torch.no_grad():
        actual_o, actual_state = tileops_fn(q, k, v, g, beta)
        ref_o, ref_state = ref_fn(q, k, v, g, beta)
        torch.cuda.synchronize()

    o_allclose = bool(
        torch.allclose(actual_o.float(), ref_o.float(), atol=args.atol, rtol=args.rtol)
    )
    state_allclose = bool(
        torch.allclose(
            actual_state.float(), ref_state.float(), atol=args.atol, rtol=args.rtol
        )
    )
    row = {
        "row_kind": "production_surface_correctness_metrics",
        "variant": "tileops_final_dispatch",
        "reference_variant": "ref_fla_051",
        "shape_label": label,
        "shape": _shape_dict(shape_args),
        "input_hash": input_hash,
        "input_contract": input_meta,
        "correctness": {
            "status": "pass" if o_allclose and state_allclose else "fail",
            "pass_rule": "torch.allclose(actual.float(), reference.float(), atol=atol, rtol=rtol)",
            "comparison_dtype": "float32",
            "tolerance": {"atol": args.atol, "rtol": args.rtol},
            "o_allclose": o_allclose,
            "final_state_allclose": state_allclose,
            "o": _error_metrics(actual_o, ref_o, sample_limit=args.sample_limit),
            "final_state": _error_metrics(
                actual_state,
                ref_state,
                sample_limit=args.sample_limit,
            ),
        },
        "environment": _environment(
            "tileops_final_dispatch",
            args.production_root,
            tileops_used_code_root,
        ),
        "reference_used_code_root": reference_used_code_root,
        "reference_identity": _fla_reference_identity(),
        "code": {
            "collector": str(Path(__file__).resolve()),
            "collector_commit": _git_commit(REPO_ROOT),
            "collector_dirty": _git_dirty(REPO_ROOT),
        },
    }

    del actual_o, actual_state, ref_o, ref_state, tensors
    torch.cuda.empty_cache()
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _fmt_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Production Surface Correctness Metrics",
        "",
        "This correctness-only refresh compares `tileops_final_dispatch` against",
        "the recorded FLA reference for the five headline synthetic-input",
        "serving shapes. The JSONL records whether that reference is a verified",
        "package version such as `flash-linear-attention==0.5.1` or a vendored",
        "source snapshot. It does not collect latency.",
        "",
        "| Shape | Status | o max_abs | o p99_abs | o mean_abs | o L2 rel | final_state max_abs | final_state p99_abs | final_state L2 rel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        o = row["correctness"]["o"]
        s = row["correctness"]["final_state"]
        lines.append(
            "| {shape} | {status} | {o_max} | {o_p99} | {o_mean} | {o_l2} | {s_max} | {s_p99} | {s_l2} |".format(
                shape=row["shape_label"],
                status=row["correctness"]["status"],
                o_max=_fmt_float(o["max_abs"]),
                o_p99=_fmt_float(o["p99_abs"]),
                o_mean=_fmt_float(o["mean_abs"]),
                o_l2=_fmt_float(o["norm_rel_l2"]),
                s_max=_fmt_float(s["max_abs"]),
                s_p99=_fmt_float(s["p99_abs"]),
                s_l2=_fmt_float(s["norm_rel_l2"]),
            )
        )

    first = rows[0]
    lines.extend(
        [
            "",
            "Contract:",
            "",
            f"- tolerance: `atol=rtol={first['correctness']['tolerance']['atol']}`",
            "- comparison dtype: fp32",
            "- p95/p99 for very large tensors use deterministic even-stride sampling;",
            "  each JSONL row records `quantile_method`, `sample_size`, and `numel`.",
            "- `max_rel_diagnostic` is retained in JSONL but should be interpreted",
            "  together with absolute error because near-zero references can dominate",
            "  relative error.",
            "",
            "Input hashes:",
            "",
            "| Shape | Input hash |",
            "| --- | --- |",
        ]
    )
    for row in rows:
        lines.append(f"| {row['shape_label']} | `{row['input_hash']}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "results" / "production_surface_correctness_metrics_20260708.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=THIS_DIR / "summaries" / "production_surface_correctness_metrics_20260708.md",
    )
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PR1596_ROOT)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gpu-contract", default="GPU4/H200")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10_000_000,
        help="Maximum elements used for p95/p99 sampling per tensor.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=[label for label, _seq, _heads in HEADLINE_SHAPES],
        help="Optional shape label; repeatable. Defaults to all headline shapes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.shape) if args.shape else None
    rows = []
    for label, seq_len, heads in HEADLINE_SHAPES:
        if selected is not None and label not in selected:
            continue
        print(f"[collect] {label}", flush=True)
        rows.append(_collect_one(args, label, seq_len, heads))
    _write_jsonl(args.output, rows)
    _write_summary(args.summary, rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
