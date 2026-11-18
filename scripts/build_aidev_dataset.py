"""
scripts/build_aidev_dataset.py — Phase 8B Dataset Builder Script

Usage:
    # 25-PR pilot (validate pipeline end-to-end)
    python scripts/build_aidev_dataset.py --pilot

    # Full dataset (~3000 PRs)
    python scripts/build_aidev_dataset.py --target 3000

    # Development cap (300 PRs)
    python scripts/build_aidev_dataset.py --target 300

    # Patch-only mode (no checkout, only Cat A+B features)
    python scripts/build_aidev_dataset.py --target 3000 --no-checkout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer.ml.dataset_builder import AIDevDatasetBuilder
from analyzer.ml.labels import compute_class_balance
from analyzer.ml.preprocessing import run_quality_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AIDev ML dataset for Phase 8B risk model."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode (25 PRs only, for pipeline validation).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3000,
        help="Target number of successfully extracted PRs (default: 3000).",
    )
    parser.add_argument(
        "--no-checkout",
        action="store_true",
        help="Disable repository checkout (only Cat A+B patch-based features).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for Parquet dataset.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output filename (default: aidev_features.parquet or aidev_pilot.parquet).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--git-timeout",
        type=int,
        default=120,
        help="Git operation timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--clone-depth",
        type=int,
        default=50,
        help="Git clone depth (default: 50).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=5,
        help="Max repos to keep on disk simultaneously (default: 5).",
    )
    args = parser.parse_args()

    # Validate config
    if args.pilot and args.target != 3000:
        logger.warning(
            "--pilot flag overrides --target. Running with 25 PRs."
        )

    pilot_size = 25
    target_size = args.target
    pilot_mode = args.pilot

    # Output file naming
    output_dir = args.output_dir
    if args.output_file:
        output_path = output_dir / args.output_file
    elif pilot_mode:
        output_path = output_dir / "aidev_pilot.parquet"
    else:
        output_path = output_dir / f"aidev_features_{target_size}.parquet"

    logger.info("=" * 60)
    logger.info("Phase 8B — AIDev Dataset Builder")
    logger.info(f"  Mode: {'PILOT (25 PRs)' if pilot_mode else f'FULL ({target_size} PRs)'}")
    logger.info(f"  Checkout: {'ENABLED' if not args.no_checkout else 'DISABLED (patch-only)'}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Random seed: {args.seed}")
    logger.info("=" * 60)

    builder = AIDevDatasetBuilder(
        output_dir=output_dir,
        checkout_enabled=not args.no_checkout,
        random_seed=args.seed,
        git_timeout=args.git_timeout,
        clone_depth=args.clone_depth,
        max_repos_on_disk=args.max_repos,
    )

    t0 = time.time()
    try:
        rows = builder.build(
            target_size=target_size,
            pilot_mode=pilot_mode,
            pilot_size=pilot_size,
        )
    finally:
        # Always attempt cleanup on exit (unless user wants to inspect)
        pass  # cleanup() called separately if needed

    elapsed = time.time() - t0
    logger.info(f"Extraction completed in {elapsed:.1f}s")
    logger.info(builder.counters.summary())

    if not rows:
        logger.error("No rows extracted. Dataset is empty. Investigate filters/connectivity.")
        sys.exit(1)

    # Save to Parquet
    saved_path = builder.save_parquet(rows, output_path)
    logger.info(f"Dataset saved: {saved_path} ({len(rows)} rows)")

    # Quality report
    row_dicts = [r.to_dict() for r in rows]
    quality = run_quality_checks(row_dicts)
    logger.info(quality.summary())

    # Class balance
    labels = [int(r.high_risk) for r in rows]
    balance = compute_class_balance(labels)
    logger.info(
        f"Class balance: {balance['n_positive']} HIGH RISK / "
        f"{balance['n_negative']} LOW RISK "
        f"(pos_frac={balance['pos_fraction']:.3f})"
    )
    logger.info(f"Suggested class weight for training: 1:{balance['suggested_weight_ratio']}")

    # Save extraction statistics
    stats_path = output_dir / (output_path.stem + "_stats.json")
    stats = {
        "n_rows": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "pilot_mode": pilot_mode,
        "target_size": target_size,
        "checkout_enabled": not args.no_checkout,
        "random_seed": args.seed,
        "counters": {
            "considered": builder.counters.considered,
            "excluded_language": builder.counters.excluded_language,
            "excluded_license": builder.counters.excluded_license,
            "excluded_fork": builder.counters.excluded_fork,
            "excluded_no_patch": builder.counters.excluded_no_patch,
            "excluded_no_review": builder.counters.excluded_no_review,
            "excluded_not_closed": builder.counters.excluded_not_closed,
            "excluded_revert": builder.counters.excluded_revert,
            "checkout_failed": builder.counters.checkout_failed,
            "analysis_failed": builder.counters.analysis_failed,
            "successfully_extracted": builder.counters.successfully_extracted,
        },
        "class_balance": balance,
        "quality_passed": quality.passed,
        "quality_warnings": quality.warnings,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved: {stats_path}")

    if pilot_mode:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PILOT COMPLETE. If pipeline looks correct, run:")
        logger.info("  python scripts/build_aidev_dataset.py --target 3000")
        logger.info("=" * 60)
    else:
        if len(rows) < target_size * 0.5:
            logger.warning(
                f"Only {len(rows)} PRs extracted vs target {target_size}. "
                "Review filter counters above for blockers."
            )


if __name__ == "__main__":
    main()
