#!/usr/bin/env python3
"""Placeholder component collector for the GDN prefill blog ladder.

The full component collection needs explicit event boundaries for each variant.
Until those adapters exist, this script writes diagnostic-only JSONL rows rather
than mixing component timings into the full-op ladder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from variants import VARIANTS, get_variant


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "component_breakdown.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    for variant_id in args.variant:
        spec = get_variant(variant_id)
        rows.append({
            "variant_id": variant_id,
            "evidence_lane": "component",
            "decision": "diagnostic_only" if variant_id in VARIANTS else "unavailable",
            "component_breakdown": {
                "status": "not_collected",
                "reason": (
                    "component event boundaries are not wired into the experiment-only "
                    "harness yet; keep this separate from full-op ladder rows"
                ),
                "required_components": [
                    "chunk_local_cumsum",
                    "A_producer",
                    "recompute_w_u_or_equivalent",
                    "CP_preprocess_corrected_starts",
                    "fused_replay_output",
                    "output_only",
                    "other",
                ],
            },
            "code_pointer": spec.code_pointer,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} component placeholder row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
