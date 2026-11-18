"""Development-only ApacheJIT structural-feature feasibility pilot.

This runner deliberately reuses ``analyzer.ml.dataset_builder._run_full_analysis``
and the repository object cache.  It never changes labels, reads final-test
rows into the cohort, or promotes a pilot model into the runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyzer.languages.detector import LanguageDetector
from analyzer.languages.registry import default_registry
from analyzer.ml.dataset_builder import (
    _repository_lease,
    _run_full_analysis,
    _run_git,
)
from dataset.config import ALL_FEATURES


RAW = ROOT / "data/raw/apachejit/apachejit_total.csv"
SPLIT_REPORT = ROOT / "reports/risk_model/metrics.json"
OUT = ROOT / "reports/risk_model/apachejit_structural_pilot"
CACHE = ROOT / "data/cache/apachejit_structural_pilot"
SEED = 20260913
PILOT_SIZE = 500
DEFAULT_PROJECTS = ("apache/camel", "apache/flink", "apache/hbase", "apache/ignite")
APACHE_FEATURES = ("la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp")
METRICS = ("precision", "recall", "f1", "roc_auc", "pr_auc")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_key(project: str, commit_id: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{project}:{commit_id}".encode()).hexdigest()


def load_development_rows(raw_path: Path = RAW, split_report: Path = SPLIT_REPORT) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    required = {"commit_id", "project", "buggy", "author_date", *APACHE_FEATURES}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"ApacheJIT columns missing: {sorted(missing)}")
    strategy = json.loads(split_report.read_text(encoding="utf-8"))["split_strategy"]
    project_split = {project: split_name for split_name, projects in strategy["projects"].items() for project in projects}
    raw["split"] = raw["project"].map(project_split)
    if raw["split"].isna().any():
        raise ValueError("Preserved project split does not cover every ApacheJIT project")
    labels = raw["buggy"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    if labels.isna().any():
        raise ValueError("ApacheJIT label is not Boolean")
    raw["buggy"] = labels.astype("int8")
    return raw.loc[raw["split"].isin(["train", "validation"])].copy()


def choose_pilot(
    development: pd.DataFrame,
    *,
    projects: Iterable[str] = DEFAULT_PROJECTS,
    size: int = PILOT_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Deterministically sample within project and label, preserving prevalence."""
    selected_projects = tuple(projects)
    if not 3 <= len(selected_projects) <= 5:
        raise ValueError("Pilot must cover 3-5 repositories")
    if size < len(selected_projects) * 2:
        raise ValueError("Pilot is too small to include both labels per project")
    if set(development.loc[development.project.isin(selected_projects), "split"]) - {"train", "validation"}:
        raise AssertionError("Final-test rows entered pilot candidates")

    base, remainder = divmod(size, len(selected_projects))
    pieces: list[pd.DataFrame] = []
    for project_index, project in enumerate(selected_projects):
        quota = base + int(project_index < remainder)
        pool = development.loc[development.project.eq(project)].copy()
        if pool.empty or pool["buggy"].nunique() != 2:
            raise ValueError(f"{project} cannot supply both labels")
        prevalence = float(pool["buggy"].mean())
        positive_quota = min(max(round(quota * prevalence), 1), quota - 1)
        label_quotas = {1: positive_quota, 0: quota - positive_quota}
        for label, label_quota in label_quotas.items():
            candidates = pool.loc[pool.buggy.eq(label)].copy()
            if len(candidates) < label_quota:
                raise ValueError(f"{project} has only {len(candidates)} label={label} rows")
            candidates["_selection_key"] = [
                _stable_key(project, commit_id, seed) for commit_id in candidates.commit_id.astype(str)
            ]
            pieces.append(candidates.sort_values("_selection_key").head(label_quota))
    pilot = pd.concat(pieces, ignore_index=True).sort_values(["project", "_selection_key"]).reset_index(drop=True)
    if len(pilot) != size or pilot.commit_id.duplicated().any():
        raise AssertionError("Pilot size/identity contract failed")
    if set(pilot["split"]) - {"train", "validation"}:
        raise AssertionError("Closed final-test partition was touched")
    if pilot.groupby("project")["buggy"].nunique().min() != 2:
        raise AssertionError("Every selected repository must include both labels")
    pilot["sample_order"] = np.arange(1, len(pilot) + 1)
    return pilot.drop(columns="_selection_key")


def _language_composition(paths: Iterable[str]) -> tuple[dict[str, int], list[str], list[str]]:
    detector = LanguageDetector()
    counts: Counter[str] = Counter()
    supported: list[str] = []
    unsupported: list[str] = []
    for path in paths:
        language = detector.detect(str(path)).language
        counts[language] += 1
        (supported if default_registry.is_supported(language) else unsupported).append(str(path))
    return dict(sorted(counts.items())), sorted(supported), sorted(unsupported)


def _empty_result(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        "sample_order": int(row["sample_order"]), "project": str(row["project"]),
        "commit_sha": str(row["commit_id"]), "parent_sha": None, "split": str(row["split"]),
        "apachejit_label": int(row["buggy"]), "checkout_success": False,
        "analysis_status": "not_started", "analysis_failure_reason": None,
        "language_composition": "{}", "supported_changed_files": 0,
        "unsupported_changed_files": 0, "analysis_seconds": None,
        "full_42_features": False, "partial_features": False,
    }
    result.update({name: row.get(name) for name in APACHE_FEATURES})
    result.update({name: np.nan for name in ALL_FEATURES})
    return result


def process_commit(
    row: dict[str, Any], *, cache_dir: Path = CACHE, git_timeout: int = 180,
    analyze_supported: bool = True,
) -> dict[str, Any]:
    from repository.git_object_cache import create_worktree, ensure_bare_repo, fetch, has_commit

    result = _empty_result(row)
    project, commit_sha = result["project"], result["commit_sha"]
    started = time.monotonic()
    repo_dir: Path | None = None
    bare: Path | None = None
    try:
        with _repository_lease(cache_dir, project, git_timeout):
            bare, _ = ensure_bare_repo(cache_dir, project, timeout=git_timeout)
            if not has_commit(bare, commit_sha, timeout=30):
                ok, error = fetch(bare, f"https://github.com/{project}.git", commit_sha, timeout=git_timeout, depth=2)
                if not ok:
                    result["analysis_status"] = "checkout_failed"
                    result["analysis_failure_reason"] = "HEAD_COMMIT_UNAVAILABLE: " + error
                    return result
        parent_ok, parent_sha = _run_git(["rev-parse", f"{commit_sha}^"], cwd=str(bare), timeout=git_timeout)
        if not parent_ok:
            result["analysis_status"] = "checkout_failed"
            result["analysis_failure_reason"] = "exact parent/base SHA unavailable: " + parent_sha
            return result
        object_ok, object_type = _run_git(["cat-file", "-t", parent_sha], cwd=str(bare), timeout=git_timeout)
        if not object_ok or object_type.strip() != "commit":
            result["analysis_status"] = "checkout_failed"
            result["analysis_failure_reason"] = "parent/base object is not an exact commit"
            return result
        # Disable rename detection: path-only eligibility needs tree objects,
        # not blob similarity scans (which hydrate a filtered object cache).
        names_ok, names_text = _run_git(["diff", "--no-renames", "--name-only", parent_sha, commit_sha], cwd=str(bare), timeout=git_timeout)
        if not names_ok:
            result["analysis_status"] = "checkout_failed"
            result["analysis_failure_reason"] = "changed-file discovery failed: " + names_text
            return result
        changed_paths = [name for name in names_text.splitlines() if name.strip()]
        composition, supported, unsupported = _language_composition(changed_paths)
        result.update({
            "parent_sha": parent_sha.strip(), "checkout_success": True,
            "language_composition": _json(composition),
            "supported_changed_files": len(supported), "unsupported_changed_files": len(unsupported),
        })
        if not supported:
            languages = ", ".join(sorted(composition)) or "none"
            result["analysis_status"] = "analysis_failed"
            result["analysis_failure_reason"] = f"NO_SUPPORTED_SOURCE_CHANGE (detected: {languages})"
            return result
        if not analyze_supported:
            result["analysis_status"] = "eligible_pending"
            result["analysis_failure_reason"] = None
            return result

        # Materialize patch blobs only after the cheap tree/path gate. Most
        # ApacheJIT rows are Java/Scala, so doing this earlier would download
        # unsupported source merely to reach the same explicit failure.
        diff_ok, diff = _run_git(
            ["diff", "--find-renames", "--no-ext-diff", parent_sha, commit_sha],
            cwd=str(bare), timeout=git_timeout,
        )
        if not diff_ok:
            result["analysis_status"] = "checkout_failed"
            result["analysis_failure_reason"] = "exact BASE..HEAD diff failed: " + diff
            return result
        if not diff.strip():
            result["analysis_status"] = "analysis_failed"
            result["analysis_failure_reason"] = "empty exact BASE..HEAD diff"
            return result

        with _repository_lease(cache_dir, project, git_timeout):
            worktree_ok, repo_dir, worktree_diag = create_worktree(
                bare, cache_dir, project, int(row["sample_order"]), "head", commit_sha,
                timeout=git_timeout, fetch_remote=f"https://github.com/{project}.git",
            )
        if not worktree_ok or repo_dir is None:
            result["analysis_status"] = "checkout_failed"
            result["checkout_success"] = False
            result["analysis_failure_reason"] = "WORKTREE_FAILED: " + _json(worktree_diag)
            return result

        diagnostics: dict[str, Any] = {}
        analysis_started = time.monotonic()
        features = _run_full_analysis(
            repo_dir, diff, parent_sha.strip(), diagnostics=diagnostics,
            max_total_runtime=600, max_dependency_neighborhood_files=5000,
            max_parsed_files=5000, max_graph_nodes=100000,
        )
        result["analysis_seconds"] = round(time.monotonic() - analysis_started, 6)
        if features is None:
            result["analysis_status"] = "analysis_failed"
            result["analysis_failure_reason"] = diagnostics.get("failure_reason") or diagnostics.get("safety_failure") or "production analyzer returned no features"
            return result
        ordered = {name: features.get(name, np.nan) for name in ALL_FEATURES}
        result.update(ordered)
        finite = [math.isfinite(float(value)) for value in ordered.values() if value is not None]
        full = len(finite) == len(ALL_FEATURES) and all(finite)
        partial = any(finite) and not full
        result["full_42_features"], result["partial_features"] = full, partial
        result["analysis_status"] = "full" if full else "partial" if partial else "analysis_failed"
        result["analysis_failure_reason"] = None if (full or partial) else "all structural fields unavailable"
        return result
    except Exception as exc:
        result["analysis_status"] = "checkout_failed" if not result["checkout_success"] else "analysis_failed"
        result["analysis_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if repo_dir is not None and bare is not None:
            _run_git(["worktree", "remove", "--force", str(repo_dir)], cwd=str(bare), timeout=git_timeout)
            _run_git(["worktree", "prune"], cwd=str(bare), timeout=30)
        result["total_seconds"] = round(time.monotonic() - started, 6)


def _write_empty(path: Path, columns: Iterable[str]) -> None:
    pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def prefetch_project_commits(rows: list[dict[str, Any]], cache_dir: Path, git_timeout: int) -> dict[str, str]:
    """Fetch each project's exact pilot SHAs in one filtered negotiation."""
    from repository.git_object_cache import ensure_bare_repo

    outcomes: dict[str, str] = {}

    def one(project: str, project_rows: list[dict[str, Any]]) -> tuple[str, str]:
        try:
            with _repository_lease(cache_dir, project, git_timeout):
                bare, _ = ensure_bare_repo(cache_dir, project, timeout=git_timeout)
                missing = [str(row["commit_id"]) for row in project_rows]
                ok, message = _run_git(
                    ["fetch", "--no-tags", "--depth", "2", "--filter=blob:none",
                     f"https://github.com/{project}.git", *missing],
                    cwd=str(bare), timeout=git_timeout,
                )
            return project, "ok" if ok else "batch_fetch_failed: " + message[-500:]
        except Exception as exc:
            return project, f"batch_fetch_failed: {type(exc).__name__}: {exc}"

    by_project = {project: group for project, group in pd.DataFrame(rows).groupby("project", sort=True)}
    with ThreadPoolExecutor(max_workers=len(by_project)) as pool:
        futures = [pool.submit(one, project, group.to_dict("records")) for project, group in by_project.items()]
        for future in as_completed(futures):
            project, outcome = future.result()
            outcomes[project] = outcome
            print(f"prefetch {project}: {outcome}", flush=True)
    return outcomes


def _quality_and_correlations(usable: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    quality_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    nonredundant: list[str] = []
    for feature in ALL_FEATURES:
        values = pd.to_numeric(usable[feature], errors="coerce")
        available = values.dropna()
        top_share = float(available.value_counts(normalize=True).iloc[0]) if len(available) else np.nan
        row = {
            "feature": feature, "available_count": int(values.notna().sum()),
            "missing_rate": float(values.isna().mean()) if len(values) else 1.0,
            "variance": float(available.var(ddof=0)) if len(available) else np.nan,
            "unique_values": int(available.nunique()), "constant": bool(available.nunique() <= 1),
            "near_constant_99pct": bool(len(available) and top_share >= .99), "largest_value_share": top_share,
        }
        max_abs = 0.0
        for apache in APACHE_FEATURES:
            paired = pd.concat([values, pd.to_numeric(usable[apache], errors="coerce")], axis=1).dropna()
            corr = float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")) if len(paired) >= 3 and paired.iloc[:, 0].nunique() > 1 and paired.iloc[:, 1].nunique() > 1 else np.nan
            correlation_rows.append({"structural_feature": feature, "apachejit_feature": apache, "spearman_correlation": corr, "paired_rows": len(paired)})
            if math.isfinite(corr):
                max_abs = max(max_abs, abs(corr))
        row["max_abs_spearman_with_apachejit"] = max_abs if len(available) else np.nan
        row["nonredundant_candidate"] = bool(len(available) >= max(20, .8 * len(usable)) and available.nunique() > 1 and top_share < .99 and max_abs < .8)
        if row["nonredundant_candidate"]:
            nonredundant.append(feature)
        quality_rows.append(row)
    quality = pd.DataFrame(quality_rows)
    correlations = pd.DataFrame(correlation_rows)
    quality.to_csv(out_dir / "structural_feature_quality.csv", index=False)
    correlations.to_csv(out_dir / "feature_correlation.csv", index=False)
    return quality, correlations, nonredundant


def _metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    predicted = (probability >= .5).astype(int)
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
    }


def _models(usable: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline

    sets = {
        "A_apachejit_safe": list(APACHE_FEATURES),
        "B_structural_42": list(ALL_FEATURES),
        "C_combined": list(APACHE_FEATURES) + list(ALL_FEATURES),
    }
    y = usable.apachejit_label.to_numpy(dtype=int)
    groups = usable.project.astype(str).to_numpy()
    results: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    logo = LeaveOneGroupOut()
    for fold, (train_idx, validation_idx) in enumerate(logo.split(usable, y, groups), 1):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[validation_idx])) < 2:
            continue
        for model_name, columns in sets.items():
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=1),
            )
            model.fit(usable.iloc[train_idx][columns], y[train_idx])
            probability = model.predict_proba(usable.iloc[validation_idx][columns])[:, 1]
            metrics = _metric_values(y[validation_idx], probability)
            results.append({"scope": "fold", "fold": fold, "held_out_project": groups[validation_idx][0], "model": model_name, "rows": len(validation_idx), **metrics})
            if model_name == "C_combined":
                perm = permutation_importance(model, usable.iloc[validation_idx][columns], y[validation_idx], scoring="average_precision", n_repeats=15, random_state=SEED)
                for feature, mean, std in zip(columns, perm.importances_mean, perm.importances_std):
                    importances.append({"fold": fold, "held_out_project": groups[validation_idx][0], "model": model_name, "diagnostic": "permutation_pr_auc", "feature": feature, "importance_mean": mean, "importance_std": std})
    fold_frame = pd.DataFrame(results)
    summaries = []
    for model_name, group in fold_frame.groupby("model"):
        summaries.append({"scope": "mean", "fold": "all", "held_out_project": "project-aware mean", "model": model_name, "rows": len(usable), **{metric: float(group[metric].mean()) for metric in METRICS}})
        summaries.append({"scope": "std", "fold": "all", "held_out_project": "project-aware std", "model": model_name, "rows": len(usable), **{metric: float(group[metric].std(ddof=0)) for metric in METRICS}})
    model_results = pd.concat([fold_frame, pd.DataFrame(summaries)], ignore_index=True)
    importance = pd.DataFrame(importances)
    model_results.to_csv(out_dir / "pilot_model_results.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    pivots = fold_frame.pivot(index="held_out_project", columns="model", values=list(METRICS))
    stable = {}
    for metric in ("roc_auc", "pr_auc", "f1"):
        if (metric, "A_apachejit_safe") in pivots and (metric, "C_combined") in pivots:
            deltas = pivots[(metric, "C_combined")] - pivots[(metric, "A_apachejit_safe")]
            stable[metric] = {"folds_improved": int((deltas > 0).sum()), "folds_total": int(len(deltas)), "mean_delta": float(deltas.mean())}
    return model_results, importance, stable


def run(args: argparse.Namespace) -> None:
    if OUT.exists() and any(OUT.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing pilot artifacts: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    development = load_development_rows(args.raw, args.split_report)
    cohort = choose_pilot(development, projects=args.projects, size=args.size, seed=args.seed)
    # Interleave repositories so each worker can use a distinct object cache.
    rows = []
    grouped = [group.to_dict("records") for _, group in cohort.groupby("project", sort=True)]
    for offset in range(max(map(len, grouped))):
        rows.extend(group[offset] for group in grouped if offset < len(group))
    prefetch_project_commits(rows, args.cache, args.git_timeout)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process_commit, row, cache_dir=args.cache, git_timeout=args.git_timeout, analyze_supported=False): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 10 == 0 or len(results) == len(rows):
                checkout = sum(bool(item["checkout_success"]) for item in results)
                usable = sum(item["analysis_status"] in {"full", "partial"} for item in results)
                print(f"completed={len(results)}/{len(rows)} checkout={checkout} usable={usable}", flush=True)
    checkout_ceiling = sum(bool(item["checkout_success"]) for item in results) / max(len(results), 1)
    eligible = [item for item in results if item["analysis_status"] == "eligible_pending"]
    usable_ceiling = len(eligible) / max(len(results), 1)
    preanalysis_gate = checkout_ceiling >= .90 and usable_ceiling >= .80
    if preanalysis_gate:
        by_order = {int(item["sample_order"]): item for item in results}
        source_rows = {int(row["sample_order"]): row for row in rows}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(process_commit, source_rows[int(item["sample_order"])], cache_dir=args.cache,
                            git_timeout=args.git_timeout, analyze_supported=True): item
                for item in eligible
            }
            for future in as_completed(futures):
                analyzed = future.result()
                by_order[int(analyzed["sample_order"])] = analyzed
        results = list(by_order.values())
    else:
        reason = (
            f"ANALYSIS_NOT_RUN_FEASIBILITY_STOP: only {len(eligible)}/{len(results)} "
            f"({usable_ceiling:.2%}) commits contain a supported changed file; 80% usable yield is impossible"
        )
        for item in eligible:
            item["analysis_status"] = "analysis_failed"
            item["analysis_failure_reason"] = reason
    manifest = pd.DataFrame(results).sort_values("sample_order")
    ordered = [c for c in manifest if c not in ALL_FEATURES] + list(ALL_FEATURES)
    manifest = manifest[ordered]
    manifest.to_csv(OUT / "pilot_manifest.csv", index=False)
    checkout_failures = manifest.loc[~manifest.checkout_success, ["sample_order", "project", "commit_sha", "analysis_failure_reason"]]
    checkout_failures.to_csv(OUT / "checkout_failures.csv", index=False)
    analysis_failures = manifest.loc[manifest.checkout_success & ~manifest.analysis_status.isin(["full", "partial"]), ["sample_order", "project", "commit_sha", "parent_sha", "language_composition", "analysis_failure_reason"]]
    analysis_failures.to_csv(OUT / "analysis_failures.csv", index=False)

    sampled = len(manifest)
    checkout_count = int(manifest.checkout_success.sum())
    usable = manifest.loc[manifest.analysis_status.isin(["full", "partial"])].copy()
    analysis_times = pd.to_numeric(manifest.analysis_seconds, errors="coerce").dropna()
    disk_bytes = sum(path.stat().st_size for path in args.cache.rglob("*") if path.is_file())
    checkout_rate = checkout_count / sampled if sampled else 0.0
    usable_rate = len(usable) / sampled if sampled else 0.0
    full_count = int(manifest.full_42_features.sum())
    partial_count = int(manifest.partial_features.sum())
    language_distribution: Counter[str] = Counter()
    for payload in manifest.language_composition:
        language_distribution.update(json.loads(payload or "{}"))

    if len(usable):
        _, _, nonredundant = _quality_and_correlations(usable, OUT)
    else:
        pd.DataFrame([{
            "feature": feature, "available_count": 0, "missing_rate": 1.0,
            "variance": np.nan, "unique_values": 0, "constant": True,
            "near_constant_99pct": False, "largest_value_share": np.nan,
            "max_abs_spearman_with_apachejit": np.nan,
            "nonredundant_candidate": False,
        } for feature in ALL_FEATURES]).to_csv(OUT / "structural_feature_quality.csv", index=False)
        pd.DataFrame([{
            "structural_feature": structural, "apachejit_feature": apache,
            "spearman_correlation": np.nan, "paired_rows": 0,
        } for structural in ALL_FEATURES for apache in APACHE_FEATURES]).to_csv(
            OUT / "feature_correlation.csv", index=False
        )
        nonredundant = []

    gate_passed = checkout_rate >= .90 and usable_rate >= .80
    stable: dict[str, Any] = {}
    if gate_passed and len(usable) >= 400 and usable.project.nunique() >= 3:
        _, _, stable = _models(usable, OUT)
    else:
        _write_empty(OUT / "pilot_model_results.csv", ["scope", "fold", "held_out_project", "model", "rows", *METRICS])
        _write_empty(OUT / "feature_importance.csv", ["fold", "held_out_project", "model", "diagnostic", "feature", "importance_mean", "importance_std"])

    consistent_gain = bool(stable) and all(item["folds_improved"] == item["folds_total"] and item["mean_delta"] > 0 for item in stable.values())
    scale = gate_passed and len(usable) >= 400 and consistent_gain
    median = float(analysis_times.median()) if len(analysis_times) else None
    p95 = float(analysis_times.quantile(.95)) if len(analysis_times) else None
    seconds_per_commit = float(pd.to_numeric(manifest.total_seconds, errors="coerce").median())
    projections = {str(count): {"serial_hours": seconds_per_commit * count / 3600, "storage_gib": disk_bytes / max(sampled, 1) * count / 1024**3} for count in (5000, 10000, 15000)}
    lines = [
        "# ApacheJIT structural feature pilot", "",
        "## Protocol", "",
        f"A deterministic, label-stratified sample of {sampled} commits was drawn only from preserved training/development projects: {', '.join(args.projects)}. No final-test project was read into feature development. Labels and natural within-project prevalence were preserved. Exact Git commit and parent objects were used; patches were not used to reconstruct source. The existing production analyzer stack was retained without rewriting it. A path/language preflight found {int((manifest.supported_changed_files > 0).sum())} potentially analyzable commits ({float((manifest.supported_changed_files > 0).mean()):.2%}); because the maximum possible usable yield was already below 80%, the required stop rule prevented further analyzer execution.", "",
        "## Feasibility", "",
        f"- Total sampled: {sampled}", f"- Checkout success: {checkout_count} ({checkout_rate:.2%})", f"- Checkout failure: {sampled-checkout_count}",
        f"- Analysis success: {len(usable)} ({usable_rate:.2%})", f"- Analysis failure: {sampled-len(usable)}", f"- Full 42-feature rows: {full_count}", f"- Partial-feature rows: {partial_count}",
        f"- Changed-file language distribution: {_json(dict(language_distribution))}", f"- Median analysis time: {median if median is not None else 'unavailable'} seconds", f"- p95 analysis time: {p95 if p95 is not None else 'unavailable'} seconds", f"- Repository/object-cache disk usage: {disk_bytes / 1024**3:.3f} GiB", "",
        f"Feasibility gate: **{'PASS' if gate_passed else 'STOP'}**. Required checkout >=90% and usable structural extraction >=80%.", "",
        "## Final decision", "",
        f"1. Can ApacheJIT commits be reliably reconstructed? **{'Yes' if checkout_rate >= .90 else 'No'}** ({checkout_rate:.2%} exact checkout success).",
        f"2. What percentage yields usable structural features? **{usable_rate:.2%}**.",
        f"3. Which languages are successfully analyzed? **{', '.join(sorted({lang for payload, status in zip(manifest.language_composition, manifest.analysis_status) if status in {'full','partial'} for lang in json.loads(payload)})) or 'None in the usable cohort'}**.",
        f"4. Which structural features contain nonredundant information? **{', '.join(nonredundant) if nonredundant else 'None established by this pilot'}**.",
        f"5. Does ApacheJIT + structural outperform ApacheJIT-only? **{'Yes in the development pilot' if consistent_gain else 'Not established'}**.",
        f"6. Is the gain stable across projects/folds? **{'Yes' if consistent_gain else 'No or not evaluated because the feasibility/sample gate failed'}**.",
        f"7. Should we scale to 5k? **{'Yes' if scale else 'No'}**.",
        f"8. Estimated cost/time to scale: {_json(projections)}. These are linear pilot extrapolations, not a scale authorization.",
        "9. Architecture incompatibilities found: Java and Scala are detected but have no registered production adapter. ApacheJIT is dominated by Java-family projects; those files are explicitly unavailable and are never sent through the Python adapter. A diagnostic attempt also showed that the production runtime guard does not interrupt repository indexing because its guard is checked only after the index walk.", "",
        "No runtime model was changed. No large-scale extraction was started.", "",
        "APACHEJIT_STRUCTURAL_PILOT_COMPLETE",
    ]
    (OUT / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("APACHEJIT_STRUCTURAL_PILOT_COMPLETE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--split-report", type=Path, default=SPLIT_REPORT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    parser.add_argument("--size", type=int, default=PILOT_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--git-timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
