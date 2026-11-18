"""Offline Phase 7C AIDev audit entry point.

Input is a JSON object with ``repositories``, ``pull_requests`` and optional
``file_details`` arrays from a pinned local AIDev snapshot. It never downloads
data, changes source data, labels outcomes, or extracts features.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dataset.phase7c_aidev_population_audit import audit_aidev_population


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline AIDev candidate-population audit")
    parser.add_argument("input", type=Path, help="Pinned local JSON source snapshot")
    parser.add_argument("output", type=Path, help="Audit report JSON path")
    parser.add_argument("--cutoff", required=True, help="Immutable audit cutoff in ISO-8601 UTC")
    parser.add_argument("--source-revision", required=True, help="Pinned AIDev source revision")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")).astimezone(UTC)
    report = audit_aidev_population(payload.get("repositories", ()), payload.get("pull_requests", ()),
                                    payload.get("file_details", ()), audited_at=cutoff,
                                    source_identity={"source_revision": args.source_revision,
                                                     "source_snapshot": str(args.input)})
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
