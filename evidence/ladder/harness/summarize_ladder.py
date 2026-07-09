#!/usr/bin/env python3
"""Generate a small markdown summary from ladder JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "summaries" / "ladder_summary.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _latency(row: dict[str, Any]) -> Any:
    latency = row.get("latency")
    if isinstance(latency, dict):
        return latency.get("latency_ms", latency.get("status"))
    return latency


def _correctness(row: dict[str, Any]) -> Any:
    correctness = row.get("correctness")
    if isinstance(correctness, dict):
        return correctness.get("status")
    return correctness


def _root_kind(row: dict[str, Any]) -> Any:
    return row.get("environment", {}).get("used_code_root", {}).get("kind")


def _root_module(row: dict[str, Any]) -> Any:
    return row.get("environment", {}).get("used_code_root", {}).get("module_file")


def _ref_kind(row: dict[str, Any]) -> Any:
    return row.get("correctness", {}).get("reference_used_code_root", {}).get("kind")


def _short_warnings(row: dict[str, Any]) -> str:
    warnings = row.get("publication_warnings", []) or []
    if not warnings:
        return ""
    if any("FLA 0.5.1 package version is not verified" in warning for warning in warnings):
        return "FLA version unverified"
    return "; ".join(str(warning) for warning in warnings)


def _shape(row: dict[str, Any]) -> str:
    shape = row.get("shape", {})
    return (
        f"B={shape.get('B')},T={shape.get('T')},H={shape.get('H')},"
        f"DK={shape.get('DK')},DV={shape.get('DV')},chunk={shape.get('chunk_size')}"
    )


def _write_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.append(
        "| variant | role | lane | causal | formal accepted | correct | latency_ms | "
        "ref verified | caveat | used root | ref root |"
    )
    lines.append("|---|---|---|---|---|---|---:|---|---|---|---|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row.get("variant_id"),
                    row.get("publication_role"),
                    row.get("evidence_lane"),
                    row.get("causal_ladder_eligible"),
                    row.get("publication_eligible"),
                    _correctness(row),
                    _latency(row),
                    row.get("reference_version_verified"),
                    _short_warnings(row),
                    _root_kind(row),
                    _ref_kind(row),
                )
            )
            + " |"
        )


def _warnings(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    seen = set()
    for row in rows:
        for warning in row.get("publication_warnings", []) or []:
            key = (row.get("variant_id"), warning)
            if key not in seen:
                seen.add(key)
                out.append(f"- {row.get('variant_id')}: {warning}")
    return out


def _abi_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("abi_equivalence", {}).get("status") not in (None, "not_applicable")
    ]


def render(rows: list[dict[str, Any]], *, source: Path) -> str:
    formal_accepted = [row for row in rows if row.get("publication_eligible") is True]
    causal = [
        row for row in formal_accepted
        if row.get("causal_ladder_eligible") is True
        and row.get("publication_role") == "causal_ladder_row"
    ]
    final_candidates = [
        row for row in formal_accepted
        if row.get("publication_role") == "final_candidate"
    ]
    external_anchors = [
        row for row in formal_accepted
        if row.get("evidence_lane") == "external_anchor"
    ]

    first = rows[0] if rows else {}
    lines = [
        "# GDN Prefill Ladder Summary",
        "",
        f"- Source JSONL: `{source}`",
        f"- Rows: {len(rows)}",
        f"- Publication-eligible evidence rows: {len(formal_accepted)}",
        f"- Controlled causal rows: {len(causal)}",
        f"- Final candidate rows: {len(final_candidates)}",
        f"- External anchor rows: {len(external_anchors)}",
        f"- Shape: {_shape(first) if rows else 'N/A'}",
        f"- Input artifact: `{first.get('input_artifact')}`" if first.get("input_artifact") else "- Input artifact: N/A",
        f"- Input hash: `{first.get('input_hash')}`" if first.get("input_hash") else "- Input hash: N/A",
        (
            "- Evidence lane note: `final_candidate` rows and `causal_ladder_row` rows "
            "are reported separately; this summary does not compute causal speedup "
            "between them."
        ),
        "",
        "## All Rows",
        "",
    ]
    _write_table(lines, rows)

    lines.extend(["", "## Publication-Eligible Evidence Rows", ""])
    _write_table(lines, formal_accepted)

    lines.extend(["", "## Controlled Causal Ladder Rows", ""])
    _write_table(lines, causal)

    lines.extend(["", "## Final Candidate Rows", ""])
    _write_table(lines, final_candidates)

    lines.extend(["", "## External Anchor Rows", ""])
    _write_table(lines, external_anchors)

    abi_rows = _abi_rows(rows)
    if abi_rows:
        lines.extend(["", "## ABI / A-Producer Equivalence", ""])
        lines.append(
            "| variant | abi status | A comparison | A allclose | A max_abs | A max_rel | note |"
        )
        lines.append("|---|---|---|---|---:|---:|---|")
        for row in abi_rows:
            abi = row.get("abi_equivalence", {})
            a_cmp = (
                abi.get("generic_vs_blocksolve_a")
                or abi.get("blocked_inverse_vs_generic_a")
                or {}
            )
            meta = row.get("abi_metadata", {})
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row.get("variant_id"),
                        abi.get("status"),
                        abi.get("comparison"),
                        a_cmp.get("allclose"),
                        a_cmp.get("max_abs"),
                        a_cmp.get("max_rel"),
                        meta.get("producer_extra_work_note"),
                    )
                )
                + " |"
            )

    warnings = _warnings(rows)
    if warnings:
        lines.extend([
            "",
            "## Warnings",
            "",
            "These warnings are also summarized in the main tables' caveat column.",
            "",
            *warnings,
        ])

    lines.extend(["", "## Code Source Audit", ""])
    lines.append("| variant | used module | root match | reference module |")
    lines.append("|---|---|---|---|")
    for row in rows:
        ref_module = row.get("correctness", {}).get("reference_used_code_root", {}).get("module_file")
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row.get("variant_id"),
                    _root_module(row),
                    row.get("environment", {}).get("used_code_root", {}).get("root_match"),
                    ref_module,
                )
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    text = render(rows, source=args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
