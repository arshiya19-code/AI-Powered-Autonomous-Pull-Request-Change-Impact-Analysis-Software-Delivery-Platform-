"""Rehabilitate and audit ApacheJIT without mutating existing artifacts.

The protocol is intentionally fixed: the prior project split is inherited,
all threshold/feature/transform decisions use development data, and the
inherited test labels are read only after those decisions are frozen in
memory.  Outputs live in new rehabilitation-only directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import skew, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/apachejit/apachejit_total.csv"
PRIOR_SPLIT = ROOT / "data/processed/risk_model/apachejit_catboost_processed.parquet"
DERIVED = ROOT / "data/processed/risk_model/apachejit_rehabilitation"
OUT = ROOT / "reports/risk_model/apachejit_rehabilitation"
MODEL_OUT = ROOT / "models/risk_model/apachejit_rehabilitation"
SEED = 20260913

ID = ["commit_id", "project", "author_date"]
LABEL = "buggy"
FEATURES = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
COUNT_FEATURES = ["la", "ld", "nf", "nd", "ns", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
ELIGIBILITY_COUNTS = [100, 250, 500, 1000, 2000]
ELIGIBILITY_SPANS = [30, 90, 180, 365]
METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def load() -> pd.DataFrame:
    raw = pd.read_csv(RAW)
    prior = pd.read_parquet(PRIOR_SPLIT)[["commit_id", "split"]]
    if prior["commit_id"].duplicated().any() or len(prior) != len(raw):
        raise ValueError("The preserved split cannot be mapped one-to-one to raw rows")
    frame = raw.merge(prior, on="commit_id", how="left", validate="one_to_one")
    if frame["split"].isna().any():
        raise ValueError("Some raw commits are absent from the preserved split")
    frame[LABEL] = frame[LABEL].astype(int)
    return frame


def data_quality(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    parsed = pd.to_datetime(frame["author_date"], unit="s", utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    invalid_time = parsed.isna() | (parsed < pd.Timestamp("1990-01-01", tz="UTC")) | (parsed > now)
    nonfinite_by_feature = {c: int((~np.isfinite(numeric[c])).sum()) for c in FEATURES}
    negative_by_feature = {c: int((numeric[c] < 0).sum()) for c in FEATURES}
    # nf/nd/ns are independently defined published metrics.  In particular,
    # subsystem grouping is not a strict child of directory grouping (root
    # files are a common counterexample), so ns > nd is not corruption.
    logically_impossible = (
        (numeric["nf"] < 1) | (numeric["nd"] < 1) | (numeric["ns"] < 1)
        | (numeric["nd"] > numeric["nf"]) | (numeric["ns"] > numeric["nf"])
    )
    identity_groups = frame.groupby("commit_id", dropna=False)
    duplicate_identity_rows = frame[frame.duplicated("commit_id", keep=False)].copy()
    conflicting_ids = identity_groups[LABEL].nunique().loc[lambda x: x > 1].index
    conflicts = frame[frame["commit_id"].isin(conflicting_ids)].copy()
    exact_duplicate_rows = int(frame.duplicated(keep=False).sum())
    predictor_groups = frame.groupby(FEATURES, dropna=False)[LABEL].agg(["size", "nunique", "sum"]).reset_index()
    contradictory_vectors = predictor_groups[predictor_groups["nunique"] > 1]
    contradictory_mask = frame.merge(contradictory_vectors[FEATURES], on=FEATURES, how="left", indicator=True)["_merge"].eq("both").to_numpy()
    zero_information = numeric.fillna(0).eq(0).all(axis=1)
    invalid_required = (~np.isfinite(numeric)).any(axis=1)
    corrupted = invalid_time | invalid_required | logically_impossible

    # Objective policy: drop all exact duplicate copies after the first; remove
    # entire conflicting identity groups; remove objectively invalid rows.
    exact_drop = frame.duplicated(keep="first")
    conflict_drop = frame["commit_id"].isin(conflicting_ids)
    removal = exact_drop | conflict_drop | corrupted
    clean = frame.loc[~removal].copy()
    clean["author_datetime_utc"] = parsed.loc[~removal]

    distinct = numeric.nunique(dropna=False)
    near_constant = {}
    for c in FEATURES:
        share = float(numeric[c].value_counts(dropna=False, normalize=True).iloc[0])
        if share >= 0.99:
            near_constant[c] = share
    extremes = {}
    for c in FEATURES:
        vals = numeric[c].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = vals.quantile([0.25, 0.75])
        fence = q3 + 10 * (q3 - q1)
        extremes[c] = {"ten_iqr_upper_fence": float(fence), "rows_above_fence": int((vals > fence).sum()), "maximum": float(vals.max())}

    report = {
        "source": {"path": str(RAW), "rows": len(frame), "sha256": sha256(RAW)},
        "preserved_split": {"path": str(PRIOR_SPLIT), "rows": len(pd.read_parquet(PRIOR_SPLIT)), "sha256": sha256(PRIOR_SPLIT)},
        "duplicate_commit_identity_rows": len(duplicate_identity_rows),
        "duplicate_commit_identity_groups": int(duplicate_identity_rows["commit_id"].nunique()),
        "exact_duplicate_rows_including_first_occurrences": exact_duplicate_rows,
        "exact_duplicate_rows_removed_after_first": int(exact_drop.sum()),
        "conflicting_duplicate_identity_groups": len(conflicting_ids),
        "conflicting_duplicate_identity_rows": len(conflicts),
        "missing_predictors": {c: int(numeric[c].isna().sum()) for c in FEATURES},
        "nan_predictors": {c: int(np.isnan(numeric[c].to_numpy(dtype=float)).sum()) for c in FEATURES},
        "nonfinite_predictors": nonfinite_by_feature,
        "negative_metrics": negative_by_feature,
        "invalid_timestamps": int(invalid_time.sum()),
        "logically_impossible_rows": int(logically_impossible.sum()),
        "zero_information_rows": int(zero_information.sum()),
        "constant_predictors": distinct[distinct <= 1].index.tolist(),
        "near_constant_predictors_99pct": near_constant,
        "extreme_values_diagnostic_only": extremes,
        "exact_contradictory_predictor_groups": len(contradictory_vectors),
        "exact_contradictory_predictor_rows": int(contradictory_mask.sum()),
        "genuinely_corrupted_or_invalid_rows": int(corrupted.sum()),
        "clean_v1_rows": len(clean),
    }
    return clean, conflicts, report


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    keys = ["project", "author_date"]
    buckets = result.groupby(keys, sort=True)[LABEL].agg(bucket_rows="size", bucket_pos="sum").reset_index()
    buckets["bucket_neg"] = buckets["bucket_rows"] - buckets["bucket_pos"]
    for source, target in [("bucket_rows", "prior_project_commits"), ("bucket_pos", "prior_project_positive_count"), ("bucket_neg", "prior_project_negative_count")]:
        buckets[target] = buckets.groupby("project")[source].cumsum() - buckets[source]
    first = buckets.groupby("project")["author_date"].transform("min")
    buckets["history_span_days"] = np.where(
        buckets["prior_project_commits"] > 0,
        (buckets["author_date"] - first) / 86400.0,
        0.0,
    )
    return result.merge(buckets[keys + ["prior_project_commits", "prior_project_positive_count", "prior_project_negative_count", "history_span_days"]], on=keys, how="left", validate="many_to_one")


def metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "brier": float(brier_score_loss(y, p)),
    }


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    rows = []
    for threshold in np.linspace(0.02, 0.98, 193):
        m = metric_row(y, p, float(threshold))
        score = np.mean([m["precision"], m["recall"], m["f1"]]) - 0.15 * abs(m["precision"] - m["recall"])
        rows.append((score, m["f1"], -abs(threshold - 0.5), float(threshold)))
    return max(rows)[-1]


def objective(m: dict[str, float]) -> float:
    return float(np.mean([m[k] for k in METRICS]) - 0.10 * abs(m["precision"] - m["recall"]) - 0.05 * m["brier"])


def make_model(y: np.ndarray, iterations: int = 220) -> CatBoostClassifier:
    negatives, positives = int((y == 0).sum()), int((y == 1).sum())
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        loss_function="Logloss",
        eval_metric="PRAUC",
        class_weights=[1.0, negatives / max(positives, 1)],
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def chronological_folds(frame: pd.DataFrame, n: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = frame.sort_values(["author_date", "commit_id"]).index.to_numpy()
    cuts = np.linspace(0, len(ordered), n + 3, dtype=int)
    folds = []
    for i in range(n):
        train_idx = ordered[: cuts[i + 2]]
        valid_idx = ordered[cuts[i + 2] : cuts[i + 3]]
        if len(valid_idx) and frame.loc[train_idx, LABEL].nunique() == 2 and frame.loc[valid_idx, LABEL].nunique() == 2:
            folds.append((train_idx, valid_idx))
    if len(folds) < 2:
        raise ValueError("Insufficient class support for chronological folds")
    return folds


@dataclass
class Transform:
    name: str
    columns: list[str]
    lower: pd.Series | None = None
    upper: pd.Series | None = None

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.columns].astype(float).copy()
        if self.name == "log1p":
            return np.log1p(x.clip(lower=0))
        if self.lower is not None and self.upper is not None:
            return x.clip(lower=self.lower, upper=self.upper, axis=1)
        return x


def fit_transform(name: str, train: pd.DataFrame, columns: list[str]) -> Transform:
    if name == "winsor_1_99":
        return Transform(name, columns, train[columns].quantile(0.01), train[columns].quantile(0.99))
    if name == "winsor_0.5_99.5":
        return Transform(name, columns, train[columns].quantile(0.005), train[columns].quantile(0.995))
    return Transform(name, columns)


def overlap_report(frame: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler(quantile_range=(25, 75)).fit(train[FEATURES])
    train_x = scaler.transform(train[FEATURES])
    # Training-only definition of "very similar": the lower decile of nearest
    # non-self same-class distances, with a small exact-equivalence floor.
    class_thresholds = []
    for cls in [0, 1]:
        x = train_x[train[LABEL].to_numpy() == cls]
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(x)
        distances = nn.kneighbors(x, return_distance=True)[0][:, 1] / math.sqrt(len(FEATURES))
        class_thresholds.append(float(np.quantile(distances, 0.10)))
    similarity = max(float(np.mean(class_thresholds)), 1e-9)
    rows = []
    for split_name, sample in [("train", train), ("validation", frame[frame.split.eq("validation")]), ("test_diagnostic_only", frame[frame.split.eq("test")])]:
        sx = scaler.transform(sample[FEATURES])
        for cls, other in [(1, 0), (0, 1)]:
            mask = sample[LABEL].to_numpy() == cls
            reference = train_x[train[LABEL].to_numpy() == other]
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(reference)
            dist = nn.kneighbors(sx[mask], return_distance=True)[0][:, 0] / math.sqrt(len(FEATURES))
            rows.append({
                "split": split_name,
                "source_class": "positive" if cls else "negative",
                "opposite_neighbor_class": "clean" if other == 0 else "positive",
                "rows": int(mask.sum()),
                "training_fitted_similarity_threshold": similarity,
                "percent_with_very_similar_opposite_neighbor": 100 * float((dist <= similarity).mean()),
                "nearest_opposite_distance_p1": float(np.quantile(dist, 0.01)),
                "nearest_opposite_distance_p50": float(np.quantile(dist, 0.50)),
                "nearest_opposite_distance_p99": float(np.quantile(dist, 0.99)),
            })
    return pd.DataFrame(rows)


def distribution_audit(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = train[FEATURES]
    for project, group in train.groupby("project"):
        rest = train.loc[~train.project.eq(project)]
        for c in FEATURES:
            values = group[c].astype(float)
            q1, q3 = values.quantile([0.25, 0.75])
            scale = float(pooled[c].quantile(0.75) - pooled[c].quantile(0.25)) or 1.0
            rows.append({
                "project": project, "feature": c, "rows": len(group),
                "median": float(values.median()), "iqr": float(q3 - q1),
                "p1": float(values.quantile(0.01)), "p5": float(values.quantile(0.05)),
                "p50": float(values.quantile(0.50)), "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "skew": float(skew(values, bias=False)) if values.nunique() > 1 else 0.0,
                "missing_rate": float(values.isna().mean()),
                "normalized_wasserstein_vs_other_training_projects": float(wasserstein_distance(values, rest[c])) / scale,
            })
    return pd.DataFrame(rows)


def project_regimes(frame: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    def summaries(source: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for project, g in source.groupby("project"):
            span_days = max((g.author_date.max() - g.author_date.min()) / 86400.0, 1.0)
            rows.append({"project": project, "typical_churn": float((g.la + g.ld).median()), "typical_file_count": float(g.nf.median()), "typical_entropy": float(g.ent.median()), "commit_frequency_per_day": len(g) / span_days, "history_length_days": span_days})
        return pd.DataFrame(rows).set_index("project")
    all_summary = summaries(frame)
    train_summary = all_summary.loc[sorted(train.project.unique())]
    scaler = StandardScaler().fit(train_summary)
    z = scaler.transform(train_summary)
    candidates = []
    for k in range(2, min(4, len(train_summary) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=25).fit(z)
        candidates.append((silhouette_score(z, km.labels_), k, km))
    score, k, model = max(candidates, key=lambda x: x[0])
    labels = model.predict(scaler.transform(all_summary))
    output = all_summary.reset_index()
    output["regime_id"] = labels
    output["fit_role"] = np.where(output.project.isin(train_summary.index), "training_fit", "assigned_without_outcomes")
    return output, {"selected_k": k, "training_silhouette": score, "selection": "maximum silhouette for k=2..4 using training-project predictor summaries only"}


def screen_history(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int]]:
    model = make_model(train[LABEL].to_numpy(), iterations=220)
    model.fit(train[FEATURES], train[LABEL])
    train_p = model.predict_proba(train[FEATURES])[:, 1]
    threshold = select_threshold(train[LABEL].to_numpy(), train_p)
    vp = model.predict_proba(validation[FEATURES])[:, 1]
    rows = []
    for count in ELIGIBILITY_COUNTS:
        for span in ELIGIBILITY_SPANS:
            mask = (validation.prior_project_commits >= count) & (validation.history_span_days >= span)
            train_mask = (train.prior_project_commits >= count) & (train.history_span_days >= span)
            if mask.sum() < 100 or validation.loc[mask, LABEL].nunique() < 2:
                continue
            metrics = metric_row(validation.loc[mask, LABEL].to_numpy(), vp[mask], threshold)
            coverage = float(mask.mean())
            tradeoff = objective(metrics) + 0.10 * coverage
            rows.append({"prior_commits": count, "minimum_history_span_days": span, "training_rows": int(train_mask.sum()), "validation_rows": int(mask.sum()), "validation_coverage": coverage, "positive_rate": float(validation.loc[mask, LABEL].mean()), "fixed_global_threshold": threshold, "development_tradeoff": tradeoff, **metrics})
    result = pd.DataFrame(rows).sort_values(["development_tradeoff", "validation_coverage"], ascending=False)
    winner = result.iloc[0]
    return result, (int(winner.prior_commits), int(winner.minimum_history_span_days))


def feature_stability(development: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    rng = np.random.default_rng(SEED)
    for fold_no, (ti, vi) in enumerate(folds, 1):
        tr, va = development.loc[ti], development.loc[vi]
        model = make_model(tr[LABEL].to_numpy(), iterations=180)
        model.fit(tr[FEATURES], tr[LABEL])
        sample = va.sample(min(5000, len(va)), random_state=SEED + fold_no)
        x = sample[FEATURES]
        y = sample[LABEL].to_numpy()
        base = average_precision_score(y, model.predict_proba(x)[:, 1])
        shap_values = model.get_feature_importance(Pool(x, feature_names=FEATURES), type="ShapValues")[:, :-1]
        for j, c in enumerate(FEATURES):
            xp = x.copy()
            xp[c] = rng.permutation(xp[c].to_numpy())
            drop = base - average_precision_score(y, model.predict_proba(xp)[:, 1])
            abs_by_project = pd.Series(np.abs(shap_values[:, j])).groupby(sample.project.to_numpy()).sum()
            dominant = float(abs_by_project.max() / max(abs_by_project.sum(), 1e-12))
            rows.append({"fold": fold_no, "feature": c, "permutation_pr_auc_drop": drop, "mean_abs_shap": float(np.abs(shap_values[:, j]).mean()), "mean_signed_shap": float(shap_values[:, j].mean()), "dominant_project_shap_share": dominant})
    detail = pd.DataFrame(rows)
    aggregate = []
    for c, g in detail.groupby("feature"):
        mean_perm = float(g.permutation_pr_auc_drop.mean())
        sd_perm = float(g.permutation_pr_auc_drop.std(ddof=1))
        direction_changes = int(np.sum(np.sign(g.mean_signed_shap.to_numpy()[1:]) != np.sign(g.mean_signed_shap.to_numpy()[:-1])))
        wildly_variable = bool(sd_perm > max(abs(mean_perm), 0.002))
        nonpositive = bool(mean_perm <= 0)
        project_dependent = bool(g.dominant_project_shap_share.max() >= 0.80)
        stable = not (nonpositive or wildly_variable or project_dependent)
        aggregate.append({"feature": c, "folds": len(g), "permutation_importance_mean": mean_perm, "permutation_importance_sd": sd_perm, "shap_importance_mean": float(g.mean_abs_shap.mean()), "signed_shap_mean": float(g.mean_signed_shap.mean()), "direction_changes": direction_changes, "max_dominant_project_shap_share": float(g.dominant_project_shap_share.max()), "flag_nonpositive_permutation": nonpositive, "flag_wild_variation": wildly_variable, "flag_single_project_dependence": project_dependent, "stable_candidate": stable})
    summary = pd.DataFrame(aggregate).sort_values("permutation_importance_mean", ascending=False)
    stable = summary.loc[summary.stable_candidate, "feature"].tolist()
    if len(stable) < 4:
        stable = summary.head(4).feature.tolist()
        summary["stable_candidate"] = summary.feature.isin(stable)
        summary["selection_note"] = "At least four features retained by descending mean permutation importance"
    return summary, stable


def evaluate_config(
    name: str,
    development: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
    eligibility: tuple[int, int] | None,
    transform_name: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    oof_y, oof_p, fold_rows = [], [], []
    for fold_no, (ti, vi) in enumerate(folds, 1):
        tr, va = development.loc[ti].copy(), development.loc[vi].copy()
        if eligibility:
            count, span = eligibility
            tr = tr[(tr.prior_project_commits >= count) & (tr.history_span_days >= span)]
            va = va[(va.prior_project_commits >= count) & (va.history_span_days >= span)]
        transform = fit_transform(transform_name, tr, columns)
        xtr, xva = transform.apply(tr), transform.apply(va)
        model = make_model(tr[LABEL].to_numpy(), iterations=220)
        model.fit(xtr, tr[LABEL])
        probability = model.predict_proba(xva)[:, 1]
        oof_y.append(va[LABEL].to_numpy()); oof_p.append(probability)
        fold_rows.append({"experiment": name, "evaluation_scope": "development_forward_fold", "fold": fold_no, "rows": len(va), "coverage_within_fold": len(va) / max(len(development.loc[vi]), 1), "positive_rate": float(va[LABEL].mean()), "transform": transform_name, "features": ";".join(columns)})
    y, p = np.concatenate(oof_y), np.concatenate(oof_p)
    threshold = select_threshold(y, p)
    for row, yy, pp in zip(fold_rows, oof_y, oof_p):
        row.update(metric_row(yy, pp, threshold)); row["threshold"] = threshold; row["objective"] = objective(row)
    aggregate = {"experiment": name, "evaluation_scope": "development_forward_oof", "fold": "all", "rows": len(y), "coverage_within_fold": len(y) / sum(len(development.loc[vi]) for _, vi in folds), "positive_rate": float(y.mean()), "transform": transform_name, "features": ";".join(columns), "threshold": threshold, **metric_row(y, p, threshold)}
    aggregate["objective"] = objective(aggregate)
    fold_rows.append(aggregate)
    return fold_rows, y, p


def fit_final_and_test(
    name: str,
    development: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    eligibility: tuple[int, int] | None,
    transform_name: str,
    threshold: float,
) -> tuple[dict[str, Any], CatBoostClassifier, Transform, pd.DataFrame]:
    tr, te = development.copy(), test.copy()
    if eligibility:
        count, span = eligibility
        tr = tr[(tr.prior_project_commits >= count) & (tr.history_span_days >= span)]
        te = te[(te.prior_project_commits >= count) & (te.history_span_days >= span)]
    transform = fit_transform(transform_name, tr, columns)
    model = make_model(tr[LABEL].to_numpy(), iterations=220)
    model.fit(transform.apply(tr), tr[LABEL])
    p = model.predict_proba(transform.apply(te))[:, 1]
    metrics = metric_row(te[LABEL].to_numpy(), p, threshold)
    row = {"experiment": name, "evaluation_scope": "inherited_test_once", "fold": "final", "rows": len(te), "coverage_within_fold": len(te) / len(test), "positive_rate": float(te[LABEL].mean()), "transform": transform_name, "features": ";".join(columns), "threshold": threshold, **metrics}
    row["objective"] = objective(row)
    predictions = te[["commit_id", "project", LABEL]].copy()
    predictions["probability"] = p; predictions["threshold"] = threshold
    return row, model, transform, predictions


def main() -> None:
    for directory in [DERIVED, OUT, MODEL_OUT]:
        directory.mkdir(parents=True, exist_ok=True)
    expected = [OUT / name for name in ["data_quality_report.json", "duplicate_conflicts.csv", "feature_overlap_report.csv", "history_eligibility.csv", "project_distribution_shift.csv", "feature_stability.csv", "cohort_definition.json", "cleaning_manifest.json", "coverage_results.csv", "model_results.csv", "learnability_report.md"]]
    if any(path.exists() for path in expected):
        raise FileExistsError("Rehabilitation outputs already exist; refusing to overwrite them")

    raw_hash_before, split_hash_before = sha256(RAW), sha256(PRIOR_SPLIT)
    frame = load()
    clean, conflicts, quality = data_quality(frame)
    conflicts.to_csv(OUT / "duplicate_conflicts.csv", index=False)
    save_json(OUT / "data_quality_report.json", quality)
    clean = add_history(clean)
    clean_path = DERIVED / "apachejit_clean_v1.parquet"
    clean.to_parquet(clean_path, index=False)

    train = clean[clean.split.eq("train")].copy()
    validation = clean[clean.split.eq("validation")].copy()
    test = clean[clean.split.eq("test")].copy()
    development = clean[~clean.split.eq("test")].copy()

    overlap = overlap_report(clean, train)
    overlap.to_csv(OUT / "feature_overlap_report.csv", index=False)
    distribution = distribution_audit(train)
    distribution.to_csv(OUT / "project_distribution_shift.csv", index=False)
    regimes, regime_meta = project_regimes(clean, train)
    regimes.to_csv(OUT / "project_regimes.csv", index=False)

    history, eligibility = screen_history(train, validation)
    history.to_csv(OUT / "history_eligibility.csv", index=False)
    count, span = eligibility
    clean["model_cohort"] = np.where((clean.prior_project_commits >= count) & (clean.history_span_days >= span), "MATURE_REPOSITORY", "COLD_START")
    # Rewrite only the new derived artifact to include the frozen cohort field.
    clean.to_parquet(clean_path, index=False)
    train = clean[clean.split.eq("train")].copy(); validation = clean[clean.split.eq("validation")].copy(); test = clean[clean.split.eq("test")].copy(); development = clean[~clean.split.eq("test")].copy()
    cohort = {
        "selection_scope": "development validation projects only",
        "selection_objective": "mean of Precision/Recall/F1/ROC-AUC/PR-AUC, minus imbalance and Brier penalties, plus 0.10 times coverage",
        "mature_repository": {"prior_project_commits_at_least": count, "history_span_days_at_least": span},
        "cold_start": "Any CLEAN_V1 row not meeting both mature-history conditions; remains routed to the existing global model.",
        "history_semantics": "Counts use only commits with a strictly earlier project timestamp; same-timestamp commits do not count one another.",
        "project_regime_analysis": regime_meta,
    }
    save_json(OUT / "cohort_definition.json", cohort)

    folds = chronological_folds(development)
    stability, stable = feature_stability(development, folds)
    stability.to_csv(OUT / "feature_stability.csv", index=False)

    all_results: list[dict[str, Any]] = []
    transform_screens = []
    for transform_name in ["raw", "log1p", "winsor_1_99", "winsor_0.5_99.5"]:
        rows, _, _ = evaluate_config(f"transform_screen_{transform_name}", development, folds, stable, eligibility, transform_name)
        all_results.extend(rows)
        transform_screens.append(rows[-1])
    best_transform = max(transform_screens, key=lambda r: r["objective"])["transform"]

    specs = [
        ("A_CLEAN_V1_full_cohort", FEATURES, None, "raw"),
        ("B_MATURE_REPOSITORY", FEATURES, eligibility, "raw"),
        ("C_MATURE_stable_features", stable, eligibility, "raw"),
        ("D_MATURE_stable_training_safe_transform", stable, eligibility, best_transform),
    ]
    frozen = []
    for name, columns, eligible, transform_name in specs:
        rows, _, _ = evaluate_config(name, development, folds, columns, eligible, transform_name)
        all_results.extend(rows)
        frozen.append((rows[-1]["objective"], name, columns, eligible, transform_name, rows[-1]["threshold"]))
    # Development-only model selection is complete before any inherited test label is used.
    winner = max(frozen, key=lambda item: item[0])
    test_rows = []
    winner_predictions = None
    for _, name, columns, eligible, transform_name, threshold in frozen:
        row, model, transform, predictions = fit_final_and_test(name, development, test, columns, eligible, transform_name, threshold)
        test_rows.append(row); all_results.append(row)
        if name == winner[1]:
            model.save_model(str(MODEL_OUT / "frozen_strongest_development_selected.cbm"))
            predictions.to_parquet(OUT / "strongest_cohort_test_predictions.parquet", index=False)
            winner_predictions = predictions
            save_json(MODEL_OUT / "frozen_protocol.json", {"selected_on": "development forward OOF only", "experiment": name, "features": columns, "eligibility": eligible, "transform": transform_name, "transform_lower": None if transform.lower is None else transform.lower.to_dict(), "transform_upper": None if transform.upper is None else transform.upper.to_dict(), "threshold": threshold})
    pd.DataFrame(all_results).to_csv(OUT / "model_results.csv", index=False)

    coverage_rows = []
    for split_name, part in [("all", clean), ("development", development), ("test", test)]:
        for cohort_name in ["MATURE_REPOSITORY", "COLD_START"]:
            mask = part.model_cohort.eq(cohort_name)
            coverage_rows.append({"split": split_name, "cohort": cohort_name, "rows": int(mask.sum()), "coverage": float(mask.mean()), "positive_rate": float(part.loc[mask, LABEL].mean()) if mask.any() else math.nan})
    pd.DataFrame(coverage_rows).to_csv(OUT / "coverage_results.csv", index=False)

    manifest = {
        "policy": "Remove exact duplicate copies after the first, remove every row in a conflicting commit-identity group, and remove only invalid timestamp/non-finite/logically impossible rows. Exact contradictory predictor vectors with distinct commit identities are retained and diagnosed, never relabeled.",
        "raw": {"path": str(RAW), "rows": len(frame), "sha256_before": raw_hash_before, "sha256_after": sha256(RAW)},
        "prior_split": {"path": str(PRIOR_SPLIT), "rows": len(frame), "sha256_before": split_hash_before, "sha256_after": sha256(PRIOR_SPLIT)},
        "derived": {"path": str(clean_path), "rows": len(clean), "sha256": sha256(clean_path)},
        "removed": {"genuinely_invalid_or_corrupted": quality["genuinely_corrupted_or_invalid_rows"], "exact_duplicate_copies": quality["exact_duplicate_rows_removed_after_first"], "conflicting_identity_rows": quality["conflicting_duplicate_identity_rows"]},
    }
    if raw_hash_before != sha256(RAW) or split_hash_before != sha256(PRIOR_SPLIT):
        raise AssertionError("An immutable input changed during the run")
    save_json(OUT / "cleaning_manifest.json", manifest)

    exact_rows = quality["exact_contradictory_predictor_rows"]
    exact_pct = 100 * exact_rows / len(clean)
    overlap_dev = overlap[overlap.split.eq("validation")]
    positive_overlap = float(overlap_dev.loc[overlap_dev.source_class.eq("positive"), "percent_with_very_similar_opposite_neighbor"].iloc[0])
    negative_overlap = float(overlap_dev.loc[overlap_dev.source_class.eq("negative"), "percent_with_very_similar_opposite_neighbor"].iloc[0])
    winner_test = next(row for row in test_rows if row["experiment"] == winner[1])
    mature_all_coverage = next(row for row in coverage_rows if row["split"] == "all" and row["cohort"] == "MATURE_REPOSITORY")
    winner_all_coverage = mature_all_coverage["coverage"] if winner[3] else 1.0
    unstable = stability.loc[~stability.stable_candidate, "feature"].tolist()
    a_dev = next(r for r in all_results if r["experiment"] == "A_CLEAN_V1_full_cohort" and r["evaluation_scope"] == "development_forward_oof")
    b_dev = next(r for r in all_results if r["experiment"] == "B_MATURE_REPOSITORY" and r["evaluation_scope"] == "development_forward_oof")
    baseline = {"precision": 0.6758, "recall": 0.7322, "f1": 0.7029, "roc_auc": 0.8838, "pr_auc": 0.7756, "brier": 0.1217}
    all_ninety = all(winner_test[k] >= 0.90 for k in METRICS)
    severe_shift = distribution.groupby("feature")["normalized_wasserstein_vs_other_training_projects"].max().sort_values(ascending=False)
    lines = [
        "# ApacheJIT dataset rehabilitation and learnability audit",
        "",
        "## Protocol and integrity",
        "",
        f"The immutable raw CSV contains {len(frame):,} rows across {frame.project.nunique()} projects. Its SHA-256 is `{raw_hash_before}`. The prior split artifact was also hashed before and after. Both hashes remained unchanged. CLEAN_V1 is a new derived parquet with {len(clean):,} rows.",
        "",
        "The inherited split is project-isolated (9 train, 3 validation, 3 test projects). Threshold, eligibility, feature, transform, and winning-cohort choices use development data only. Forward chronological development folds preserve the observed class ratios. The inherited test has been evaluated in prior work, so its results are useful comparison evidence, not a fresh independent confirmation.",
        "",
        "## Final decisions",
        "",
        f"1. **Genuinely corrupted/invalid records:** {quality['genuinely_corrupted_or_invalid_rows']:,}. Exact duplicate copies removed: {quality['exact_duplicate_rows_removed_after_first']:,}; conflicting-identity rows removed: {quality['conflicting_duplicate_identity_rows']:,}.",
        f"2. **Conflicting-label overlap:** {quality['conflicting_duplicate_identity_groups']:,} conflicting commit-identity groups. Separately, {quality['exact_contradictory_predictor_groups']:,} exact predictor-vector groups cover {exact_rows:,} rows ({exact_pct:.2f}% of CLEAN_V1). On validation data, {positive_overlap:.2f}% of positives have a very-similar clean training neighbor and {negative_overlap:.2f}% of negatives have a very-similar positive training neighbor under the training-fitted diagnostic threshold.",
        f"3. **Fraction with sufficient history:** {mature_all_coverage['coverage']:.2%} of CLEAN_V1 meets the selected mature-history rule; {1-mature_all_coverage['coverage']:.2%} remains cold-start fallback.",
        f"4. **Best development eligibility tradeoff:** at least {count:,} prior project commits and {span} prior history days. Selection used the documented balanced development objective plus a small coverage term, never test outcomes.",
        f"5. **Unstable features:** {', '.join(unstable) if unstable else 'none under the predeclared flags'}. Stable candidate subset: {', '.join(stable)}.",
        f"6. **Does CLEAN_V1 improve performance?** CLEAN_V1 removed {len(frame)-len(clean):,} rows, so this dataset provides {'no cleaning-driven change to measure' if len(frame)==len(clean) else 'an objective cleaned comparison'}. Its forward-development metrics are Precision {a_dev['precision']:.4f}, Recall {a_dev['recall']:.4f}, F1 {a_dev['f1']:.4f}, ROC-AUC {a_dev['roc_auc']:.4f}, PR-AUC {a_dev['pr_auc']:.4f}, Brier {a_dev['brier']:.4f}; comparison to the prior headline is not causal because the validation protocol differs.",
        f"7. **Does mature history improve performance?** On like-for-like forward development evaluation, the full mature model objective is {b_dev['objective']:.4f} versus {a_dev['objective']:.4f} for full-cohort CLEAN_V1 ({'improved' if b_dev['objective'] > a_dev['objective'] else 'not improved'}), with mature-fold coverage {b_dev['coverage_within_fold']:.2%}.",
        f"8. **Coverage of the strongest development-selected cohort:** {winner_test['coverage_within_fold']:.2%} of the inherited test and {winner_all_coverage:.2%} of the entire cleaned dataset. Experiment: {winner[1]}.",
        f"9. **Strongest cohort inherited-test metrics:** Precision {winner_test['precision']:.4f}, Recall {winner_test['recall']:.4f}, F1 {winner_test['f1']:.4f}, ROC-AUC {winner_test['roc_auc']:.4f}, PR-AUC {winner_test['pr_auc']:.4f}, Brier {winner_test['brier']:.4f}.",
        f"10. **Is >=0.90 for all five supported?** {'Yes on this inherited evaluation, but fresh-project confirmation is still required.' if all_ninety else 'No.'}",
        "11. **What prevents it?** " + ("The selected result clears the numerical target here, but the test is inherited and already observed; population-level support still requires untouched future projects." if all_ninety else f"The evidence shows predictor/label ambiguity (exact-vector overlap {exact_pct:.2f}%; validation near-opposite overlap {positive_overlap:.2f}% for positives and {negative_overlap:.2f}% for negatives), cross-project distribution shift (largest normalized Wasserstein shifts: " + ", ".join(f"{k}={v:.2f}" for k, v in severe_shift.head(4).items()) + "), unstable features, and incomplete mature-history coverage. These limit simultaneous classification and ranking performance; more CatBoost search cannot create missing causal signal."),
        "",
        "## Interpretation",
        "",
        f"Label prevalence is {clean[LABEL].mean():.2%} overall, {development[LABEL].mean():.2%} in development, and {test[LABEL].mean():.2%} in the inherited test. Natural ratios were retained. Large but valid changes were represented through raw/log1p/training-fitted winsor candidates and were never deleted merely for size. The chosen transform was `{best_transform}`.",
        "",
        f"The prior strongest headline was Precision {baseline['precision']:.4f}, Recall {baseline['recall']:.4f}, F1 {baseline['f1']:.4f}, ROC-AUC {baseline['roc_auc']:.4f}, PR-AUC {baseline['pr_auc']:.4f}, Brier {baseline['brier']:.4f}. Because this audit changes cohorts and uses forward folds, differences should be read as learnability evidence rather than a clean model-only uplift claim.",
        "",
        "Cold-start commits were not discarded: the cohort manifest explicitly routes them to the existing global model. No labels, project balance, source features, existing models, existing reports, or prior splits were changed.",
        "",
        "APACHEJIT_DATASET_REHABILITATION_COMPLETE",
    ]
    (OUT / "learnability_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("APACHEJIT_DATASET_REHABILITATION_COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
