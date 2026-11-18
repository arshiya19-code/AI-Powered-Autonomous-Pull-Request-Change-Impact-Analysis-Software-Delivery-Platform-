"""Acquire GitHub identities for, and only for, the existing 300 AIDev rows."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.phase7f_identity_acquisition import (  # noqa: E402
    AcquisitionError, GitHubClient, IdentityAcquirer, load_fixed_identities,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded acquisition for the existing 300 PR identities.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/aidev_features_300.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/aidev_300_acquisition"))
    parser.add_argument("--cutoff-at", required=True, help="Fixed UTC ISO-8601 audit cutoff, e.g. 2026-09-06T00:00:00Z")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--allow-unauthenticated", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token and not args.allow_unauthenticated:
        raise SystemExit("GITHUB_TOKEN or GH_TOKEN is required (use --allow-unauthenticated only for a deliberately rate-limited run).")
    try:
        cutoff = datetime.fromisoformat(args.cutoff_at.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("--cutoff-at must be timezone-aware ISO-8601") from exc
    try:
        identities = load_fixed_identities(args.input)
        client = GitHubClient(token, args.output_dir / "raw_github", retries=args.retries, timeout_seconds=args.timeout_seconds)
        acquirer = IdentityAcquirer(client, args.output_dir, cutoff.astimezone(UTC))
        acquirer.write_immutable_manifest(identities, args.input)
        acquirer.acquire_all(identities)
    except AcquisitionError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
