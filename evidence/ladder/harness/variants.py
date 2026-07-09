"""Experiment-only variant registry for the GDN prefill blog ladder.

This module intentionally does not modify production dispatch.  Every runnable
row must be selected by its stable variant id from this registry.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PR1596_ROOT = Path(os.environ.get("TILEOPS_GDN_PR1596_ROOT", "/home/ga/TileOPs-pr1596"))
DEFAULT_HISTORY_ROOT_BASE = Path(
    os.environ.get("TILEOPS_GDN_HISTORY_ROOT_BASE", "/home/ga/TileOPs-gdn-history")
)
HISTORICAL_PREFILL_ROOTS = {
    "local_initial_prefill_f147": DEFAULT_HISTORY_ROOT_BASE / "initial-f1472392",
    "local_prepare_specialized_00a60": DEFAULT_HISTORY_ROOT_BASE / "prepare-00a60b19",
    "local_h_tile_tuned_827": DEFAULT_HISTORY_ROOT_BASE / "htile-82707454",
    "local_bthd_wall_d09c": DEFAULT_HISTORY_ROOT_BASE / "bthdwall-d09c8f2d",
}


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    evidence_lane: str
    publication_role: str
    causal_ladder_eligible: bool
    decision: str
    status: str
    description: str
    code_pointer: str
    schedule: str
    a_producer: str
    runnable: bool
    notes: tuple[str, ...] = ()
    dispatch_metadata: dict[str, Any] = field(default_factory=dict)


class VariantUnavailable(RuntimeError):
    """Raised when a registry row is intentionally not runnable yet."""


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
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_dirty(root: Path) -> bool | None:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _path_meta(root: Path) -> dict[str, Any]:
    return {
        "path": str(root),
        "exists": root.exists(),
        "commit": _git_commit(root) if root.exists() else None,
        "dirty": _git_dirty(root) if root.exists() else None,
    }


def _fla_vendor_commit(vendor: Path) -> str | None:
    commit_file = vendor / ".git_commit.txt"
    if commit_file.exists():
        return commit_file.read_text(encoding="utf-8").strip()
    return None


def _module_file_from_object(obj: Any) -> str | None:
    module = sys.modules.get(getattr(obj, "__module__", ""))
    filename = getattr(module, "__file__", None)
    return str(Path(filename).resolve()) if filename else None


def _module_file(fn: Callable[..., Any]) -> str | None:
    return _module_file_from_object(fn)


def _under(path: str | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _fla_source_identity(
    fn: Callable[..., Any],
    *,
    import_mode: str,
    vendor: Path | None = None,
) -> dict[str, Any]:
    module_file = _module_file(fn)
    known_vendors = (
        REPO_ROOT / ".github/runner/vendor/flash-linear-attention",
        REPO_ROOT / ".github/runner/vendor/native-sparse-attention/3rdparty/flash-linear-attention",
    )
    actual_vendor = vendor
    if actual_vendor is None:
        for candidate in known_vendors:
            if _under(module_file, candidate):
                actual_vendor = candidate
                break

    package_version = _package_version("flash-linear-attention") or _package_version("fla")
    if actual_vendor is not None:
        return {
            "kind": "fla_vendor",
            "path": str(actual_vendor),
            "exists": actual_vendor.exists(),
            "commit": None,
            "dirty": None,
            "vendor_commit_file": _fla_vendor_commit(actual_vendor),
            "commit_note": "vendored FLA directory is not treated as an independent git worktree",
            "module_file": module_file,
            "import_mode": import_mode,
            "package_version": package_version,
        }

    module_path = Path(module_file).parent if module_file else Path("<unknown>")
    return {
        "kind": "fla_package",
        "path": str(module_path),
        "exists": module_file is not None,
        "commit": None,
        "dirty": None,
        "module_file": module_file,
        "import_mode": import_mode,
        "package_version": package_version,
    }


def _attach_used_code_root(
    fn: Callable[..., Any],
    used_code_root: dict[str, Any],
) -> Callable[..., Any]:
    setattr(fn, "_tileops_ladder_used_code_root", used_code_root)
    return fn


def _attach_abi_metadata(
    fn: Callable[..., Any],
    abi_metadata: dict[str, Any],
) -> Callable[..., Any]:
    setattr(fn, "_tileops_ladder_abi_metadata", abi_metadata)
    return fn


def _max_abs_rel(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def _clear_tileops_modules() -> None:
    for name in list(sys.modules):
        if name == "tileops" or name.startswith("tileops."):
            del sys.modules[name]


def _path_inside(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _activate_tileops_root(root: Path, *, competing_roots: tuple[Path, ...]) -> None:
    root = root.resolve()
    roots = tuple({candidate.resolve() for candidate in (*competing_roots, root)})
    sys.path[:] = [
        p for p in sys.path
        if p and not any(_path_inside(p, candidate) for candidate in roots)
    ]
    sys.path.insert(0, str(root))
    _clear_tileops_modules()


def _install_tileops_gdn_shims(root: Path) -> None:
    """Expose only the TileOps packages needed by historical GDN checkpoints.

    Older TileOps trees import every op/kernel from ``tileops.ops.__init__`` and
    ``tileops.kernels.__init__``.  That pulls in unrelated attention kernels
    whose TileLang imports are not compatible with the current Docker runner.
    The GDN checkpoint itself only needs the package paths below, so install
    minimal package shells and let Python load the requested GDN modules from
    disk without executing those broad package initializers.
    """
    root = root.resolve()
    package_paths = {
        "tileops": root / "tileops",
        "tileops.ops": root / "tileops/ops",
        "tileops.kernels": root / "tileops/kernels",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__file__ = str(path / "__init__.py")
        package.__package__ = name
        sys.modules[name] = package


def _tileops_source_identity(
    *,
    kind: str,
    root: Path,
    obj: Any,
    import_mode: str,
) -> dict[str, Any]:
    module_file = _module_file_from_object(obj)
    return {
        "kind": kind,
        **_path_meta(root),
        "module_file": module_file,
        "import_mode": import_mode,
        "root_match": _under(module_file, root),
    }


def _mixed_v5_source_identity(
    *,
    producer_module: str | None,
    downstream_module: str | None,
    production_root: Path,
) -> dict[str, Any]:
    producer_match = _under(producer_module, REPO_ROOT)
    downstream_match = _under(downstream_module, production_root)
    return {
        "kind": "mixed_experiment_roots",
        "path": str(Path(__file__).resolve()),
        "exists": True,
        "commit": _git_commit(REPO_ROOT),
        "dirty": _git_dirty(REPO_ROOT),
        "module_file": str(Path(__file__).resolve()),
        "import_mode": "current_repo_generic_a_plus_pr1596_cp_downstream",
        "root_match": bool(producer_match and downstream_match),
        "root_match_detail": {
            "generic_a_producer": producer_match,
            "cp_downstream": downstream_match,
        },
        "generic_a_producer_root": {
            "kind": "tileops_repo",
            **_path_meta(REPO_ROOT),
            "module_file": producer_module,
        },
        "cp_downstream_root": {
            "kind": "production_root",
            **_path_meta(production_root),
            "module_file": downstream_module,
        },
    }


VARIANTS: dict[str, VariantSpec] = {
    "ref_fla_051": VariantSpec(
        variant_id="ref_fla_051",
        evidence_lane="external_anchor",
        publication_role="correctness_oracle_and_fla_baseline",
        causal_ladder_eligible=False,
        decision="accept",
        status="runnable_if_fla_installed",
        description="FLA 0.5.1 chunk_gated_delta_rule reference, BTHD inputs.",
        code_pointer="fla.ops.gated_delta_rule.chunk_gated_delta_rule",
        schedule="FLA chunk gated delta rule",
        a_producer="FLA internal",
        runnable=True,
        notes=("Correctness oracle and FLA latency baseline; not a TileOps causal row.",),
    ),
    "generic_a_legacy": VariantSpec(
        variant_id="generic_a_legacy",
        evidence_lane="controlled_full_op",
        publication_role="causal_ladder_row",
        causal_ladder_eligible=True,
        decision="accept",
        status="implemented_current_repo",
        description="Current repo generic WY/KKT-style A producer plus legacy replay/output.",
        code_pointer="tileops.ops.GatedDeltaNetFwdOp",
        schedule="legacy chunk replay/output",
        a_producer="generic exact/KKT-style WY inverse",
        runnable=True,
        notes=(
            "Current implementation is BHSD; harness converts from canonical BTHD before timing.",
            "Forward returns S/Aw/Au training artifacts; final_state is S[:, :, -1].",
        ),
    ),
    "local_initial_prefill_f147": VariantSpec(
        variant_id="local_initial_prefill_f147",
        evidence_lane="historical_full_op",
        publication_role="historical_local_ladder_row",
        causal_ladder_eligible=False,
        decision="accept",
        status="implemented_historical_worktree",
        description="Initial correct TileOps prefill op checkpoint at commit f1472392.",
        code_pointer="/home/ga/TileOPs-gdn-history/initial-f1472392/tileops/ops/gated_deltanet.py",
        schedule="legacy prefill replay/output",
        a_producer="initial TileOps prefill A producer",
        runnable=True,
        notes=(
            "Historical full-op checkpoint for the blog story table.",
            "Uses a detached worktree and canonical BTHD inputs converted to the commit's public op layout.",
        ),
    ),
    "local_prepare_specialized_00a60": VariantSpec(
        variant_id="local_prepare_specialized_00a60",
        evidence_lane="historical_full_op",
        publication_role="historical_local_ladder_row",
        causal_ladder_eligible=False,
        decision="accept",
        status="implemented_historical_worktree",
        description="Local AKO checkpoint after specializing the GDN prefill prepare path.",
        code_pointer="/home/ga/TileOPs-gdn-history/prepare-00a60b19/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py",
        schedule="legacy prefill replay/output with specialized prepare",
        a_producer="specialized generic/KKT-style prepare",
        runnable=True,
        notes=(
            "Historical full-op checkpoint; use for narrative progression, not V5/V6 controlled ABI claims.",
        ),
    ),
    "local_h_tile_tuned_827": VariantSpec(
        variant_id="local_h_tile_tuned_827",
        evidence_lane="historical_full_op",
        publication_role="historical_local_ladder_row",
        causal_ladder_eligible=False,
        decision="accept",
        status="implemented_historical_worktree",
        description="Local AKO checkpoint after tuning the h replay tile.",
        code_pointer="/home/ga/TileOPs-gdn-history/htile-82707454/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py",
        schedule="legacy prefill replay/output with tuned h tile",
        a_producer="specialized generic/KKT-style prepare",
        runnable=True,
        notes=(
            "Historical full-op checkpoint before the BTHD wall.",
            "This row captures local replay-tile tuning, not CP-split scheduling.",
        ),
    ),
    "local_bthd_wall_d09c": VariantSpec(
        variant_id="local_bthd_wall_d09c",
        evidence_lane="historical_full_op",
        publication_role="historical_local_wall_row",
        causal_ladder_eligible=False,
        decision="accept",
        status="implemented_historical_worktree",
        description="BTHD local-AKO wall checkpoint before FlashQLA-style CP split.",
        code_pointer="/home/ga/TileOPs-gdn-history/bthdwall-d09c8f2d/tileops/ops/gated_deltanet.py",
        schedule="BTHD legacy replay/output wall",
        a_producer="specialized generic/KKT-style prepare",
        runnable=True,
        notes=(
            "Historical full-op wall row; it is not a positive CP-split result.",
            "Use to close Level 2 before introducing FlashQLA in Level 3.",
        ),
    ),
    "generic_a_local_ako_best": VariantSpec(
        variant_id="generic_a_local_ako_best",
        evidence_lane="conditional_full_op",
        publication_role="conditional_causal_ladder_row",
        causal_ladder_eligible=False,
        decision="unavailable",
        status="needs_resurrection",
        description="Best local generic-A AKO end-to-end row, if one can be rerun cleanly.",
        code_pointer="/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako",
        schedule="legacy/local generic-A path",
        a_producer="generic exact/KKT-style WY inverse",
        runnable=False,
        notes=(
            "Historical component/full-op traces exist, but no clean current selectable variant was found.",
            "May enter the ladder only after end-to-end full-op correctness and latency are rerun.",
        ),
    ),
    "generic_a_direct_fused_correct": VariantSpec(
        variant_id="generic_a_direct_fused_correct",
        evidence_lane="conditional_full_op",
        publication_role="conditional_boundary_diagnostic",
        causal_ladder_eligible=False,
        decision="unavailable",
        status="not_found",
        description="Correct direct fused no-CP full-op diagnostic, if resurrected.",
        code_pointer="not found in current repo",
        schedule="direct fused replay/output without CP split",
        a_producer="generic exact/KKT-style WY inverse",
        runnable=False,
        notes=("No clean correct end-to-end implementation was found in current repo.",),
    ),
    "generic_a_direct_fused_failed": VariantSpec(
        variant_id="generic_a_direct_fused_failed",
        evidence_lane="negative_diagnostic",
        publication_role="negative_diagnostic",
        causal_ladder_eligible=False,
        decision="diagnostic_only",
        status="historical_diagnostic_only",
        description="Failed or rejected direct-fusion candidates.",
        code_pointer=(
            "/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/"
            "notes/restart_checkpoint_20260616.md"
        ),
        schedule="direct fused replay/output without CP split",
        a_producer="generic exact/KKT-style WY inverse",
        runnable=False,
        notes=(
            "Historical notes reject fused h+output diagnostics; this is not a performance milestone.",
            "Keep as negative diagnostic unless a correct direct-fused full-op row is rerun.",
        ),
    ),
    "flashqla_public_tl018": VariantSpec(
        variant_id="flashqla_public_tl018",
        evidence_lane="external_anchor",
        publication_role="external_anchor",
        causal_ladder_eligible=False,
        decision="unavailable",
        status="external_env_required",
        description="Public FlashQLA TL0.1.8 external schedule/performance anchor.",
        code_pointer="external FlashQLA TL0.1.8 environment",
        schedule="public FlashQLA CP-split replay/output",
        a_producer="FlashQLA public implementation",
        runnable=False,
        notes=("Must stay outside the controlled TileOps causal ladder.",),
    ),
    "flashqla_port_current_tl": VariantSpec(
        variant_id="flashqla_port_current_tl",
        evidence_lane="external_anchor",
        publication_role="migration_lowering_anchor",
        causal_ladder_eligible=False,
        decision="unavailable",
        status="needs_resurrection",
        description="FlashQLA skeleton migrated to current TileLang for lowering/schedule anchoring.",
        code_pointer=(
            "/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/"
            "notes/flashqla_scheduling_skeleton_20260623.md"
        ),
        schedule="FlashQLA-style CP split under current TileLang",
        a_producer="FlashQLA skeleton",
        runnable=False,
        notes=("Use for migration/lowering anchor only, not controlled TileOps causality.",),
    ),
    "tileops_owned_cp_generic_a": VariantSpec(
        variant_id="tileops_owned_cp_generic_a",
        evidence_lane="controlled_full_op",
        publication_role="causal_ladder_row",
        causal_ladder_eligible=True,
        decision="accept",
        status="implemented_experiment_only",
        description="TileOps-owned CP split with generic A producer under the fixed V5/V6 ABI.",
        code_pointer=(
            "experiments/gated_deltanet_prefill_blog_ladder/variants.py::"
            "tileops_owned_cp_generic_a adapter"
        ),
        schedule="TileOps-owned partitioned prefill / fused replay-output with generic A",
        a_producer="generic exact/KKT-style WY inverse",
        runnable=True,
        notes=(
            "Experiment-only V5 row; does not modify production dispatch.",
            "Uses current-repo generic exact/KKT-style fused_prepare_compute_w_u_tl to materialize A.",
            "Uses PR1596 CP preprocess and fused replay/output downstream with the same A handoff ABI.",
            "Generic A producer is run with g_zero to match CP downstream ABI; g_cum is passed separately.",
        ),
    ),
    "tileops_owned_cp_blocked_inverse_a": VariantSpec(
        variant_id="tileops_owned_cp_blocked_inverse_a",
        evidence_lane="controlled_full_op",
        publication_role="causal_ladder_row",
        causal_ladder_eligible=True,
        decision="accept",
        status="implemented_experiment_only",
        description="Same TileOps-owned CP split with blocked-inverse / Neumann-style A producer.",
        code_pointer=(
            "experiments/gated_deltanet_prefill_blog_ladder/variants.py::"
            "tileops_owned_cp_blocked_inverse_a adapter"
        ),
        schedule="TileOps-owned partitioned prefill / fused replay-output with blocked-inverse A",
        a_producer="blocked-inverse / Neumann-style blocksolve A",
        runnable=True,
        notes=(
            "Experiment-only V6 row; does not call the production dispatch wrapper.",
            "Uses PR1596 blocksolve A producer, CP preprocess, and fused replay/output explicitly.",
            "A producer is run with g_zero to match the same CP downstream ABI as V5.",
        ),
    ),
    "tileops_final_dispatch": VariantSpec(
        variant_id="tileops_final_dispatch",
        evidence_lane="controlled_full_op",
        publication_role="final_candidate",
        causal_ladder_eligible=False,
        decision="accept",
        status="runnable_from_pr1596_if_available",
        description="Current accepted production candidate from PR1596, selected explicitly.",
        code_pointer="/home/ga/TileOPs-pr1596/tileops.ops.GatedDeltaNetPrefillFwdOp",
        schedule="shape-aware production dispatch",
        a_producer="production dispatch A producer",
        runnable=True,
        notes=(
            "Included as final candidate/current production smoke row, not as a V5/V6 causal substitute.",
            "Requires --production-root or default /home/ga/TileOPs-pr1596 to exist.",
        ),
    ),
}


def get_variant(variant_id: str) -> VariantSpec:
    try:
        return VARIANTS[variant_id]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise VariantUnavailable(f"unknown variant {variant_id!r}; known: {known}") from exc


def _activate_external_root(root: Path) -> None:
    _activate_tileops_root(root, competing_roots=(REPO_ROOT, DEFAULT_PR1596_ROOT))


def _activate_current_root(production_root: Path | None) -> None:
    root = production_root or DEFAULT_PR1596_ROOT
    _activate_tileops_root(root, competing_roots=(REPO_ROOT, DEFAULT_PR1596_ROOT))


def _import_fla_chunk_rule() -> tuple[Callable[..., Any], dict[str, Any]]:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule, _fla_source_identity(
            chunk_gated_delta_rule,
            import_mode="installed_or_current_sys_path_import",
        )
    except ImportError:
        pass

    for vendor in (
        REPO_ROOT / ".github/runner/vendor/flash-linear-attention",
        REPO_ROOT / ".github/runner/vendor/native-sparse-attention/3rdparty/flash-linear-attention",
    ):
        if vendor.exists():
            try:
                for name in list(sys.modules):
                    if name == "fla" or name.startswith("fla."):
                        del sys.modules[name]
                sys.path.insert(0, str(vendor))
                fla_pkg = types.ModuleType("fla")
                fla_pkg.__path__ = [str(vendor / "fla")]
                ops_pkg = types.ModuleType("fla.ops")
                ops_pkg.__path__ = [str(vendor / "fla/ops")]
                sys.modules["fla"] = fla_pkg
                sys.modules["fla.ops"] = ops_pkg
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                return chunk_gated_delta_rule, _fla_source_identity(
                    chunk_gated_delta_rule,
                    import_mode="vendored_source_shim",
                    vendor=vendor,
                )
            except ImportError:
                for name in list(sys.modules):
                    if name == "fla" or name.startswith("fla."):
                        del sys.modules[name]
    raise VariantUnavailable("FLA chunk_gated_delta_rule is not importable")


def make_callable(
    variant_id: str,
    *,
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    production_root: Path | None = None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    spec = get_variant(variant_id)
    if not spec.runnable:
        raise VariantUnavailable(f"{variant_id} is {spec.status}: {spec.description}")

    if variant_id == "ref_fla_051":
        chunk_gated_delta_rule, fla_source = _import_fla_chunk_rule()

        def run_ref(q, k, v, g, beta):
            return chunk_gated_delta_rule(
                q, k, v, g, beta, scale=1.0, output_final_state=True
            )

        return _attach_used_code_root(run_ref, fla_source)

    if variant_id == "generic_a_legacy":
        root = production_root or DEFAULT_PR1596_ROOT
        _activate_current_root(production_root)
        from tileops.ops import GatedDeltaNetFwdOp

        op = GatedDeltaNetFwdOp(
            batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=False
        )

        def run_legacy(q, k, v, g, beta):
            q_bhsd = q.permute(0, 2, 1, 3).contiguous()
            k_bhsd = k.permute(0, 2, 1, 3).contiguous()
            v_bhsd = v.permute(0, 2, 1, 3).contiguous()
            g_bhsd = g.permute(0, 2, 1).contiguous()
            beta_bhsd = beta.permute(0, 2, 1).contiguous()
            o_bhsd, states, _aw, _au = op(q_bhsd, k_bhsd, v_bhsd, g_bhsd, beta_bhsd)
            o = o_bhsd.permute(0, 2, 1, 3).contiguous()
            final_state = states[:, :, -1].contiguous()
            return o, final_state

        return _attach_used_code_root(
            run_legacy,
            _tileops_source_identity(
                kind="tileops_repo",
                root=root,
                obj=GatedDeltaNetFwdOp,
                import_mode="explicit_current_repo_activation",
            ),
        )

    if variant_id in HISTORICAL_PREFILL_ROOTS:
        root = HISTORICAL_PREFILL_ROOTS[variant_id]
        if not (root / "tileops/ops/gated_deltanet.py").exists():
            raise VariantUnavailable(f"historical worktree not found or invalid: {root}")
        _activate_external_root(root)
        _install_tileops_gdn_shims(root)
        ops_gdn = importlib.import_module("tileops.ops.gated_deltanet")
        op_cls = getattr(ops_gdn, "GatedDeltaNetPrefillFwdOp")
        try:
            op = op_cls(
                batch,
                heads,
                seq_len,
                dim_k,
                dim_v,
                chunk_size,
                dtype,
                tune=False,
                layout="bthd",
            )
            op_layout = "bthd"
        except TypeError:
            op = op_cls(
                batch,
                heads,
                seq_len,
                dim_k,
                dim_v,
                chunk_size,
                dtype,
                tune=False,
            )
            op_layout = "bhtd"

        def run_historical_prefill(q, k, v, g, beta):
            if op_layout == "bthd":
                return op(q, k, v, g, beta)
            q_bhsd = q.permute(0, 2, 1, 3).contiguous()
            k_bhsd = k.permute(0, 2, 1, 3).contiguous()
            v_bhsd = v.permute(0, 2, 1, 3).contiguous()
            g_bhs = g.permute(0, 2, 1).contiguous()
            beta_bhs = beta.permute(0, 2, 1).contiguous()
            o_bhsd, final_state = op(q_bhsd, k_bhsd, v_bhsd, g_bhs, beta_bhs)
            return o_bhsd.permute(0, 2, 1, 3).contiguous(), final_state

        source_identity = _tileops_source_identity(
            kind="historical_worktree",
            root=root,
            obj=op_cls,
            import_mode=f"explicit_historical_root_activation_{op_layout}",
        )
        source_identity["historical_variant_id"] = variant_id
        source_identity["historical_commit"] = _git_commit(root)
        return _attach_used_code_root(run_historical_prefill, source_identity)

    if variant_id == "tileops_owned_cp_generic_a":
        root = production_root or DEFAULT_PR1596_ROOT
        if not (root / "tileops/kernels/gated_deltanet/gated_deltanet_prefill.py").exists():
            raise VariantUnavailable(f"production root not found or invalid: {root}")
        if chunk_size != 64 or dim_k != 128 or dim_v != 128:
            raise VariantUnavailable(
                "tileops_owned_cp_generic_a currently mirrors the PR1596 CP ABI "
                "for chunk64, DK=DV=128 only"
            )

        _activate_current_root(production_root)
        from tileops.kernels.gated_deltanet.fused_prepare_compute_w_u import (
            fused_prepare_compute_w_u_tl,
        )
        from tileops.kernels.gated_deltanet.gated_deltanet_fwd import (
            _chunk_local_cumsum,
        )

        dtype_str = str(dtype).split(".")[-1]
        generic_prepare_fn = fused_prepare_compute_w_u_tl(
            batch, heads, seq_len, chunk_size, dim_k, dim_v, dtype_str
        )(2, 256)
        producer_module = _module_file_from_object(fused_prepare_compute_w_u_tl)

        _activate_external_root(root)
        from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
            _prefill_blocksolve_A_bthd,
            _prefill_partitioned_initial_state_bthd,
        )
        from tileops.kernels.gated_deltanet.gdn_prefill import fused_gdr_fwd
        downstream_module = _module_file_from_object(fused_gdr_fwd)

        def materialize_generic_a(q, k, v, g, beta):
            del q
            k_bhsd = k.permute(0, 2, 1, 3).contiguous()
            v_bhsd = v.permute(0, 2, 1, 3).contiguous()
            beta_bhs = beta.permute(0, 2, 1).contiguous()
            # CP downstream applies g_cum itself, matching PR1596's g_zero A ABI.
            g_zero_bhs = torch.zeros_like(beta_bhs)
            aw_bhsd, au_bhsd, _w_unused, _u_unused = generic_prepare_fn(
                k_bhsd, v_bhsd, g_zero_bhs, beta_bhs
            )
            a_bthd = aw_bhsd.permute(0, 2, 1, 3).contiguous()
            return a_bthd, aw_bhsd, au_bhsd

        def run_owned_cp_generic(q, k, v, g, beta):
            a_bthd, _aw_bhsd, _au_bhsd = materialize_generic_a(q, k, v, g, beta)
            g_cum_bhs = _chunk_local_cumsum(
                g.permute(0, 2, 1).contiguous().float(), chunk_size
            ).to(g.dtype)
            g_cum_bth = g_cum_bhs.permute(0, 2, 1).contiguous()
            initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
                _prefill_partitioned_initial_state_bthd(
                    k=k,
                    v=v,
                    A=a_bthd,
                    g=g_cum_bth,
                    beta=beta,
                    chunk_size=chunk_size,
                )
            )
            o, _h, final_state = fused_gdr_fwd(
                q=q,
                k=k,
                v=v,
                a=a_bthd,
                g=g_cum_bth,
                b=beta,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=True,
                output_h=False,
                output_o=True,
                cu_seqlens=cu_seqlens,
                cp_seq_map=cp_seq_map,
                raw_cu_seqlens=raw_cu_seqlens,
                chunk_size=chunk_size,
            )
            return o, final_state.to(q.dtype)

        def collect_abi_equivalence(q, k, v, g, beta):
            a_generic, aw_bhsd, au_bhsd = materialize_generic_a(q, k, v, g, beta)
            g_zero = torch.zeros_like(g)
            a_blocksolve = _prefill_blocksolve_A_bthd(k, g_zero, beta, chunk_size)
            a_err = _max_abs_rel(a_generic, a_blocksolve)
            aw_au_err = _max_abs_rel(aw_bhsd, au_bhsd)
            return {
                "status": "collected",
                "comparison": "materialized canonical logical A",
                "canonical_logical_layout": "B,T,H,chunk_row",
                "comparison_dtype": "float32",
                "tolerance": {"atol": 5e-2, "rtol": 5e-2},
                "generic_vs_blocksolve_a": {
                    **a_err,
                    "allclose": bool(
                        torch.allclose(
                            a_generic.float(),
                            a_blocksolve.float(),
                            atol=5e-2,
                            rtol=5e-2,
                        )
                    ),
                },
                "generic_aw_vs_au_internal": {
                    **aw_au_err,
                    "allclose": bool(
                        torch.allclose(
                            aw_bhsd.float(),
                            au_bhsd.float(),
                            atol=5e-2,
                            rtol=5e-2,
                        )
                    ),
                },
                "decoded_tensors": {
                    "generic_a": {
                        "logical_shape": list(a_generic.shape),
                        "physical_strides": list(a_generic.stride()),
                        "dtype": str(a_generic.dtype).replace("torch.", ""),
                    },
                    "blocked_inverse_a": {
                        "logical_shape": list(a_blocksolve.shape),
                        "physical_strides": list(a_blocksolve.stride()),
                        "dtype": str(a_blocksolve.dtype).replace("torch.", ""),
                    },
                },
            }

        abi_metadata = {
            "status": "experiment_only_v5_adapter",
            "A_semantics": (
                "A = (I + strict_lower(diag(beta) @ K @ K^T))^-1 per chunk; "
                "g is not folded into A for the CP ABI"
            ),
            "A_logical_shape": "[B, T, H, chunk_size]",
            "physical_layout": "contiguous BTHC handoff; produced as BHSC then permuted to BTHC",
            "triangular_convention": "unit diagonal plus lower-triangular inverse rows; upper triangle zero/unused",
            "g_g_cum_convention": "generic A uses g_zero; chunk-local g_cum is passed separately to CP replay",
            "beta_folding_convention": "beta folded inside A producer and also passed to downstream CP ABI as b",
            "dtype_store_policy": "A stored in input dtype; internal accumulators are float32",
            "materialization_handoff": "materialized A tensor handed to PR1596 CP preprocess and fused_gdr_fwd",
            "CP_preprocess_API": "_prefill_partitioned_initial_state_bthd(k, v, A, g_cum, beta, chunk_size)",
            "fused_replay_output_API": "gdn_prefill.fused_gdr_fwd(q, k, v, a=A, g=g_cum, b=beta, ...)",
            "producer_extra_work_note": (
                "current generic fused_prepare_compute_w_u_tl also computes unused w/u; "
                "latency is a conservative V5 full-op row, not an A-only microbenchmark"
            ),
        }

        run_owned_cp_generic = _attach_used_code_root(
            run_owned_cp_generic,
            _mixed_v5_source_identity(
                producer_module=producer_module,
                downstream_module=downstream_module,
                production_root=root,
            ),
        )
        run_owned_cp_generic = _attach_abi_metadata(
            run_owned_cp_generic,
            abi_metadata,
        )
        setattr(
            run_owned_cp_generic,
            "_tileops_ladder_collect_abi_equivalence",
            collect_abi_equivalence,
        )
        return run_owned_cp_generic

    if variant_id == "tileops_owned_cp_blocked_inverse_a":
        root = production_root or DEFAULT_PR1596_ROOT
        if not (root / "tileops/kernels/gated_deltanet/gated_deltanet_prefill.py").exists():
            raise VariantUnavailable(f"production root not found or invalid: {root}")
        if chunk_size != 64 or dim_k != 128 or dim_v != 128:
            raise VariantUnavailable(
                "tileops_owned_cp_blocked_inverse_a currently mirrors the PR1596 CP ABI "
                "for chunk64, DK=DV=128 only"
            )

        _activate_current_root(production_root)
        from tileops.kernels.gated_deltanet.fused_prepare_compute_w_u import (
            fused_prepare_compute_w_u_tl,
        )

        dtype_str = str(dtype).split(".")[-1]
        generic_prepare_fn = fused_prepare_compute_w_u_tl(
            batch, heads, seq_len, chunk_size, dim_k, dim_v, dtype_str
        )(2, 256)
        generic_module = _module_file_from_object(fused_prepare_compute_w_u_tl)

        _activate_external_root(root)
        from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
            _prefill_blocksolve_A_bthd,
            _prefill_partitioned_initial_state_bthd,
        )
        from tileops.kernels.gated_deltanet.gdn_prefill import fused_gdr_fwd

        blocksolve_module = _module_file_from_object(_prefill_blocksolve_A_bthd)
        downstream_module = _module_file_from_object(fused_gdr_fwd)

        def chunk_local_cumsum_bth(g):
            bsz, tokens, nheads = g.shape
            return (
                g.float()
                .reshape(bsz, tokens // chunk_size, chunk_size, nheads)
                .cumsum(2)
                .reshape(bsz, tokens, nheads)
                .to(g.dtype)
            )

        def materialize_blocksolve_a(k, beta):
            g_zero = torch.zeros_like(beta)
            return _prefill_blocksolve_A_bthd(k, g_zero, beta, chunk_size).contiguous()

        def materialize_generic_a(k, v, beta):
            k_bhsd = k.permute(0, 2, 1, 3).contiguous()
            v_bhsd = v.permute(0, 2, 1, 3).contiguous()
            beta_bhs = beta.permute(0, 2, 1).contiguous()
            g_zero_bhs = torch.zeros_like(beta_bhs)
            aw_bhsd, au_bhsd, _w_unused, _u_unused = generic_prepare_fn(
                k_bhsd, v_bhsd, g_zero_bhs, beta_bhs
            )
            return aw_bhsd.permute(0, 2, 1, 3).contiguous(), aw_bhsd, au_bhsd

        def run_owned_cp_blocked(q, k, v, g, beta):
            a_bthd = materialize_blocksolve_a(k, beta)
            g_cum_bth = chunk_local_cumsum_bth(g)
            initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
                _prefill_partitioned_initial_state_bthd(
                    k=k,
                    v=v,
                    A=a_bthd,
                    g=g_cum_bth,
                    beta=beta,
                    chunk_size=chunk_size,
                )
            )
            o, _h, final_state = fused_gdr_fwd(
                q=q,
                k=k,
                v=v,
                a=a_bthd,
                g=g_cum_bth,
                b=beta,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=True,
                output_h=False,
                output_o=True,
                cu_seqlens=cu_seqlens,
                cp_seq_map=cp_seq_map,
                raw_cu_seqlens=raw_cu_seqlens,
                chunk_size=chunk_size,
            )
            return o, final_state.to(q.dtype)

        def collect_abi_equivalence(q, k, v, g, beta):
            del q, g
            a_blocked = materialize_blocksolve_a(k, beta)
            a_generic, aw_bhsd, au_bhsd = materialize_generic_a(k, v, beta)
            a_err = _max_abs_rel(a_blocked, a_generic)
            aw_au_err = _max_abs_rel(aw_bhsd, au_bhsd)
            return {
                "status": "collected",
                "comparison": "materialized canonical logical A",
                "canonical_logical_layout": "B,T,H,chunk_row",
                "comparison_dtype": "float32",
                "tolerance": {"atol": 5e-2, "rtol": 5e-2},
                "blocked_inverse_vs_generic_a": {
                    **a_err,
                    "allclose": bool(
                        torch.allclose(
                            a_blocked.float(),
                            a_generic.float(),
                            atol=5e-2,
                            rtol=5e-2,
                        )
                    ),
                },
                "generic_aw_vs_au_internal": {
                    **aw_au_err,
                    "allclose": bool(
                        torch.allclose(
                            aw_bhsd.float(),
                            au_bhsd.float(),
                            atol=5e-2,
                            rtol=5e-2,
                        )
                    ),
                },
                "decoded_tensors": {
                    "blocked_inverse_a": {
                        "logical_shape": list(a_blocked.shape),
                        "physical_strides": list(a_blocked.stride()),
                        "dtype": str(a_blocked.dtype).replace("torch.", ""),
                    },
                    "generic_a": {
                        "logical_shape": list(a_generic.shape),
                        "physical_strides": list(a_generic.stride()),
                        "dtype": str(a_generic.dtype).replace("torch.", ""),
                    },
                },
            }

        abi_metadata = {
            "status": "experiment_only_v6_adapter",
            "A_semantics": (
                "A = PR1596 blocked-inverse / blocksolve materialization for "
                "(I + strict_lower(diag(beta) @ K @ K^T))^-1 per chunk; "
                "g is not folded into A for the CP ABI"
            ),
            "A_logical_shape": "[B, T, H, chunk_size]",
            "physical_layout": "contiguous BTHC handoff produced directly by PR1596 blocksolve A",
            "triangular_convention": "unit diagonal plus lower-triangular inverse rows; upper triangle zero/unused",
            "g_g_cum_convention": "blocked-inverse A uses g_zero; chunk-local g_cum is passed separately to CP replay",
            "beta_folding_convention": "beta folded inside A producer and also passed to downstream CP ABI as b",
            "dtype_store_policy": "A stored in input dtype; internal accumulators are float32/solve dtype as in PR1596",
            "materialization_handoff": "materialized A tensor handed to PR1596 CP preprocess and fused_gdr_fwd",
            "CP_preprocess_API": "_prefill_partitioned_initial_state_bthd(k, v, A, g_cum, beta, chunk_size)",
            "fused_replay_output_API": "gdn_prefill.fused_gdr_fwd(q, k, v, a=A, g=g_cum, b=beta, ...)",
            "producer_extra_work_note": "explicit V6 adapter calls blocksolve A once and then the same CP downstream as V5",
        }

        source_identity = {
            "kind": "production_root_experiment_adapter",
            **_path_meta(root),
            "module_file": str(Path(__file__).resolve()),
            "import_mode": "explicit_blocksolve_a_plus_pr1596_cp_downstream",
            "root_match": bool(_under(blocksolve_module, root) and _under(downstream_module, root)),
            "root_match_detail": {
                "blocked_inverse_a_producer": _under(blocksolve_module, root),
                "cp_downstream": _under(downstream_module, root),
            },
            "blocked_inverse_a_producer_root": {
                "kind": "production_root",
                **_path_meta(root),
                "module_file": blocksolve_module,
            },
            "generic_a_comparison_root": {
                "kind": "tileops_repo",
                **_path_meta(REPO_ROOT),
                "module_file": generic_module,
            },
            "cp_downstream_root": {
                "kind": "production_root",
                **_path_meta(root),
                "module_file": downstream_module,
            },
        }

        run_owned_cp_blocked = _attach_used_code_root(
            run_owned_cp_blocked,
            source_identity,
        )
        run_owned_cp_blocked = _attach_abi_metadata(
            run_owned_cp_blocked,
            abi_metadata,
        )
        setattr(
            run_owned_cp_blocked,
            "_tileops_ladder_collect_abi_equivalence",
            collect_abi_equivalence,
        )
        return run_owned_cp_blocked

    if variant_id == "tileops_final_dispatch":
        root = production_root or DEFAULT_PR1596_ROOT
        if not (root / "tileops/ops/gated_deltanet.py").exists():
            raise VariantUnavailable(f"production root not found or invalid: {root}")
        _activate_external_root(root)
        ops = importlib.import_module("tileops.ops")
        op_cls = getattr(ops, "GatedDeltaNetPrefillFwdOp")
        op = op_cls(
            batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=False, layout="bthd"
        )

        def run_final(q, k, v, g, beta):
            return op(q, k, v, g, beta)

        return _attach_used_code_root(
            run_final,
            _tileops_source_identity(
                kind="production_root",
                root=root,
                obj=op_cls,
                import_mode="explicit_external_root_activation",
            ),
        )

    raise VariantUnavailable(f"{variant_id} has no callable adapter")


def variant_metadata(variant_id: str) -> dict[str, Any]:
    spec = get_variant(variant_id)
    return {
        "variant_id": spec.variant_id,
        "evidence_lane": spec.evidence_lane,
        "publication_role": spec.publication_role,
        "causal_ladder_eligible": spec.causal_ladder_eligible,
        "decision": spec.decision,
        "status": spec.status,
        "description": spec.description,
        "code_pointer": spec.code_pointer,
        "schedule": spec.schedule,
        "a_producer": spec.a_producer,
        "notes": list(spec.notes),
        "dispatch_metadata": dict(spec.dispatch_metadata),
    }


def env_override_metadata() -> dict[str, str | None]:
    keys = [
        "TILEOPS_GDN_PREFILL_CP_SPLIT",
        "TILEOPS_GDN_PREFILL_PARTITIONED",
        "TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS",
        "TILEOPS_GDN_PREFILL_CP_MAX_LOCAL_CHUNKS",
        "TILEOPS_GDN_PREFILL_FORCE_PARTITION",
        "TILEOPS_GDN_PREFILL_CP_FORCE",
    ]
    return {key: os.environ.get(key) for key in keys}
