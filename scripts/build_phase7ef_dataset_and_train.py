"""Build the Phase 7E/7F supervised AIDev dataset and train CatBoost when valid.

Usage:
    python scripts/build_phase7ef_dataset_and_train.py
    python scripts/build_phase7ef_dataset_and_train.py --expand-target 200
    python scripts/build_phase7ef_dataset_and_train.py --no-expand
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.config import SUPPORTED_GITHUB_LANGUAGES
from dataset.phase7d_outcome_run import build_outcome_artifacts
from dataset.phase7e_supervised import assemble_supervised_dataset, write_dataset_package
from dataset.phase7f_identity_acquisition import AcquisitionError, GitHubClient, IdentityAcquirer, load_identities_from_parquet
from analyzer.ml.deployment_risk import assess_sufficiency, split_matrices, train_deployment_risk_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase7ef")

IMMUTABLE_FEATURES = Path("data/processed/aidev_features_300.parquet")
IMMUTABLE_LEDGER = Path("data/processed/aidev_300_acquisition/identity_ledger.jsonl")
IMMUTABLE_LABELS = Path("data/processed/aidev_300_acquisition/outcomes/phase7d_labels_300.jsonl")
DEFAULT_CUTOFF = "2026-09-06T00:00:00Z"


def _expansion_is_complete(report: dict[str, object], target: int) -> bool:
    counters = report.get("counters", {})
    return isinstance(counters, dict) and int(counters.get("accepted", 0)) >= target


def _cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return parsed.astimezone(UTC)


def _github_client(cache_dir: Path) -> GitHubClient | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        logger.warning("No GitHub token in the environment; expansion labels will remain unavailable.")
        return None
    return GitHubClient(token, cache_dir)


def _expand(args: argparse.Namespace, cutoff: datetime) -> tuple[Path | None, Path | None, Path | None]:
    if args.no_expand or args.expand_target <= 0:
        return None, None, None
    exclude = {int(row["pr_id"]) for row in __import__("pyarrow.parquet").parquet.read_table(str(IMMUTABLE_FEATURES), columns=["pr_id"]).to_pylist()}
    expansion_dir = Path(args.expansion_dir)
    features_path = expansion_dir / "aidev_features_expansion.parquet"
    progress_path = expansion_dir / "expansion_progress_report.json"
    expansion_complete = False
    if features_path.exists() and progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            expansion_complete = int(progress.get("counters", {}).get("accepted", 0)) >= args.expand_target
        except (OSError, ValueError, TypeError):
            expansion_complete = False
    if not expansion_complete or args.force_expand:
        from dataset.phase7f_expansion import expand_feature_cohort
        client = _github_client(expansion_dir / "raw_github")
        features_path, ledger_path, labels_path, report = expand_feature_cohort(
            output_parquet=features_path,
            target_size=args.expand_target,
            exclude_pr_ids=exclude,
            cutoff_at=cutoff,
            client=client,
            workers=args.workers,
            outcome_workers=args.outcome_workers,
            candidate_limit=args.candidate_limit,
            random_seed=args.seed,
            batch_size=args.batch_size,
        )
        logger.info("Cheap-first expansion report: %s", report)
        if not _expansion_is_complete(report, args.expand_target):
            counters = report.get("counters", {})
            accepted = counters.get("accepted", 0) if isinstance(counters, dict) else 0
            raise AcquisitionError(f"expansion incomplete: accepted={accepted} target={args.expand_target}")
        return (features_path if features_path.exists() else None, ledger_path, labels_path)
    if not features_path.exists():
        return None, None, None
    ledger_path = expansion_dir / "identity_ledger.jsonl"
    client = _github_client(expansion_dir / "raw_github")
    if client and not ledger_path.exists():
        identities = load_identities_from_parquet(features_path)
        acquirer = IdentityAcquirer(client, expansion_dir, cutoff)
        if not (expansion_dir / "manifest.json").exists():
            acquirer.write_immutable_manifest(identities, features_path)
        acquirer.acquire_all(identities)
    labels_path = expansion_dir / "outcomes" / "phase7d_labels.jsonl"
    if client and ledger_path.exists() and (not labels_path.exists() or args.force_expand):
        build_outcome_artifacts(ledger_path=ledger_path, output_dir=expansion_dir / "outcomes", cutoff_at=cutoff, client=client, outcome_workers=args.outcome_workers)
    return features_path if features_path.exists() else None, ledger_path if ledger_path.exists() else None, labels_path if labels_path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7E/7F supervised AIDev dataset + CatBoost")
    parser.add_argument("--cutoff-at", default=DEFAULT_CUTOFF)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--expansion-dir", type=Path, default=Path("data/processed/aidev_expansion"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/phase7ef"))
    parser.add_argument("--expand-target", type=int, default=200, help="Additional high-quality PRs to attempt (0 disables).")
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--force-expand", action="store_true")
    parser.add_argument("--no-checkout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2, help="Bounded concurrent identity/source workers.")
    parser.add_argument("--outcome-workers", type=int, default=8, help="Bounded concurrent GitHub outcome workers.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Bound metadata candidates before outcome acquisition.")
    parser.add_argument("--batch-size", type=int, default=200, help="Bound identity, outcome, and feature batches (1-250).")
    parser.add_argument("--no-train", action="store_true", help="Build/validate the dataset without training CatBoost.")
    args = parser.parse_args()
    cutoff = _cutoff(args.cutoff_at)
    if not IMMUTABLE_FEATURES.exists():
        logger.error("Immutable 300-row feature parquet is missing: %s", IMMUTABLE_FEATURES)
        return 1
    if not IMMUTABLE_LABELS.exists():
        logger.error("Phase 7D labels are missing: %s", IMMUTABLE_LABELS)
        return 1

    feature_tables = [(IMMUTABLE_FEATURES, "aidev_features_300")]
    label_paths = [IMMUTABLE_LABELS]
    ledger_paths = [IMMUTABLE_LEDGER] if IMMUTABLE_LEDGER.exists() else []
    expansion_failed = False
    try:
        extra_features, extra_ledger, extra_labels = _expand(args, cutoff)
        if args.expand_target > 0 and not args.no_expand and not extra_features:
            raise RuntimeError("expansion returned no completed feature artifact")
    except Exception as exc:
        expansion_failed = args.expand_target > 0 and not args.no_expand
        logger.error("EXPANSION FAILED: %s", exc)
        extra_features = extra_ledger = extra_labels = None
    if extra_features:
        feature_tables.append((extra_features, "aidev_expansion"))
    if extra_ledger:
        ledger_paths.append(extra_ledger)
    if extra_labels:
        label_paths.append(extra_labels)

    result = assemble_supervised_dataset(
        feature_tables, label_paths=label_paths, ledger_paths=ledger_paths, cutoff_at=cutoff, seed=args.seed,
    )
    paths = write_dataset_package(result, args.output_dir, stem="aidev_supervised_42")
    logger.info("Supervised rows=%s excluded=%s unknown=%s", len(result.supervised_rows), len(result.excluded), result.unknown_count)

    matrices = split_matrices(result.supervised_rows)
    sufficiency = assess_sufficiency(matrices["y_train"], matrices["y_val"], matrices["y_test"])
    train_report = {"trained": False, "skipped": True} if args.no_train else train_deployment_risk_model(result.supervised_rows, output_dir=args.models_dir, random_seed=args.seed)
    summary = {
        "expansion_requested": bool(args.expand_target > 0 and not args.no_expand),
        "expansion_failed": expansion_failed,
        "expansion_succeeded": bool(args.expand_target <= 0 or args.no_expand or extra_features),
        "dataset_paths": {key: str(path) for key, path in paths.items()},
        "supervised_rows": len(result.supervised_rows),
        "label_report": result.label_report,
        "language_report": result.language_report["final_training_rows_per_language"],
        "split_report": result.split_report,
        "sufficiency": sufficiency.__dict__,
        "model_trained": train_report.get("trained"),
        "model_metadata": str(args.models_dir / "deployment_risk_model_metadata.json"),
        "supported_github_languages": sorted(SUPPORTED_GITHUB_LANGUAGES),
    }
    summary_path = args.output_dir / "aidev_supervised_42_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return 1 if expansion_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
