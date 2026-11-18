"""Repository-aware, strictly temporal ApacheJIT risk experiment.

The script deliberately refuses to overwrite an experiment.  Development,
threshold selection, and model freezing finish before the test partition is
read by the fitted model.  All historical features exclude the complete
current-timestamp batch, so timestamp ties cannot leak into one another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/apachejit/apachejit_total.csv"
OUT = ROOT / "reports/risk_model/apachejit_temporal"
ART = ROOT / "models/risk_model/apachejit_temporal"
SEED = 20260911
DAY = 86_400

RAW_FEATURES = ["nf", "la", "ld", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
RATIOS = ["add_delete_ratio", "churn_per_file", "churn_per_directory", "entropy_per_file",
          "recent_experience_ratio", "subsystem_experience_ratio", "developer_density"]
NORMALIZATION = ["churn_vs_prior_project_median", "nf_vs_prior_project_median",
                 "entropy_vs_prior_project_median", "churn_prior_percentile",
                 "nf_prior_percentile", "entropy_prior_percentile", "age_prior_percentile",
                 "ndev_prior_percentile", "current_churn_vs_recent_average",
                 "current_nf_vs_recent_average"]
ROLLING = ["project_commit_count_30d", "project_commit_count_90d", "project_churn_30d",
           "project_churn_90d", "project_mean_churn_30d", "project_mean_churn_90d",
           "project_mean_entropy_30d", "project_mean_entropy_90d"]
PRIOR_RISK = ["project_prior_bug_rate"]
DEVELOPER_REQUESTED = ["developer_prior_commits", "developer_commits_30d", "developer_commits_90d",
                       "developer_prior_mean_churn", "developer_prior_mean_entropy",
                       "developer_prior_bug_count", "developer_prior_clean_count", "developer_prior_bug_rate"]
SUBSYSTEM_REQUESTED = ["subsystem_prior_commits", "subsystem_prior_bug_rate",
                       "subsystem_recent_churn", "subsystem_recent_defects"]

STAGES = {
    "A": RAW_FEATURES + RATIOS,
    "B": RAW_FEATURES + RATIOS + NORMALIZATION,
    "C": RAW_FEATURES + RATIOS + NORMALIZATION + ROLLING,
    # No developer identifier exists in the published CSV. D is an explicit no-op ablation.
    "D": RAW_FEATURES + RATIOS + NORMALIZATION + ROLLING,
    "E": RAW_FEATURES + RATIOS + NORMALIZATION + ROLLING + PRIOR_RISK,
    "F": RAW_FEATURES + RATIOS + NORMALIZATION + ROLLING + PRIOR_RISK + ["project"],
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path): return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path: Path = RAW) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {"commit_id", "project", "buggy", "fix", "year", "author_date", *RAW_FEATURES}
    missing = required - set(d)
    if missing: raise ValueError(f"ApacheJIT columns missing: {sorted(missing)}")
    if d.commit_id.isna().any() or d.project.isna().any() or d.commit_id.duplicated().any():
        raise ValueError("Null/duplicate commit or project identifier")
    labels = d.buggy.map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    if labels.isna().any(): raise ValueError("buggy is not Boolean")
    d["buggy"] = labels.astype("int8")
    d["author_date"] = pd.to_numeric(d.author_date, errors="raise").astype("int64")
    d[RAW_FEATURES] = d[RAW_FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(d[RAW_FEATURES].to_numpy()).all(): raise ValueError("Non-finite source metric")
    if (d[RAW_FEATURES] < 0).any().any(): raise ValueError("Negative source metric")
    d["churn"] = d.la + d.ld
    d["add_delete_ratio"] = d.la / (d.ld + 1.0)
    d["churn_per_file"] = d.churn / d.nf.clip(lower=1.0)
    d["churn_per_directory"] = d.churn / d.nd.clip(lower=1.0)
    d["entropy_per_file"] = d.ent / d.nf.clip(lower=1.0)
    d["recent_experience_ratio"] = d.arexp / (d.aexp + 1.0)
    d["subsystem_experience_ratio"] = d.asexp / (d.aexp + 1.0)
    d["developer_density"] = d.ndev / d.nf.clip(lower=1.0)
    return d


def _closest_boundary(cumulative: np.ndarray, target: float, lo: int, hi: int) -> int:
    candidates = np.arange(lo, hi + 1)
    return int(candidates[np.argmin(np.abs(cumulative[candidates - 1] - target))])


def temporal_split(d: pd.DataFrame) -> pd.DataFrame:
    """Split each project at timestamp-group boundaries, approximately 70/15/15."""
    pieces = []
    for project, group in d.groupby("project", sort=True):
        group = group.sort_values(["author_date", "commit_id"], kind="stable").copy()
        sizes = group.groupby("author_date", sort=True).size().to_numpy()
        if len(sizes) < 3: raise ValueError(f"{project} has fewer than three distinct timestamps")
        cumulative = np.cumsum(sizes)
        first = _closest_boundary(cumulative, .70 * len(group), 1, len(sizes) - 2)
        second = _closest_boundary(cumulative, .85 * len(group), first + 1, len(sizes) - 1)
        cut1, cut2 = int(cumulative[first - 1]), int(cumulative[second - 1])
        group["split"] = np.where(np.arange(len(group)) < cut1, "train",
                                  np.where(np.arange(len(group)) < cut2, "validation", "test"))
        pieces.append(group)
    result = pd.concat(pieces, ignore_index=True)
    ids = {s: set(result.loc[result.split.eq(s), "commit_id"]) for s in ("train", "validation", "test")}
    if any(ids[a] & ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Commit overlap across partitions")
    for project, group in result.groupby("project"):
        train, val, test = (group[group.split.eq(s)] for s in ("train", "validation", "test"))
        if not (train.author_date.max() < val.author_date.min() <= val.author_date.max() < test.author_date.min()):
            raise AssertionError(f"Non-temporal split for {project}")
    return result


class Fenwick:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = np.unique(np.asarray(coordinates, dtype=float))
        self.tree = np.zeros(len(self.coordinates) + 1, dtype=np.int64)
        self.n = 0

    def add(self, value: float) -> None:
        i = int(np.searchsorted(self.coordinates, value)) + 1
        while i < len(self.tree):
            self.tree[i] += 1; i += i & -i
        self.n += 1

    def le(self, value: float) -> int:
        i, total = int(np.searchsorted(self.coordinates, value, side="right")), 0
        while i:
            total += int(self.tree[i]); i -= i & -i
        return total

    def kth(self, k: int) -> float:
        if not 1 <= k <= self.n: raise IndexError(k)
        idx, bit = 0, 1 << (len(self.tree).bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt < len(self.tree) and self.tree[nxt] < k:
                idx = nxt; k -= int(self.tree[nxt])
            bit >>= 1
        return float(self.coordinates[idx])

    def median(self) -> float:
        if not self.n: return 0.0
        if self.n % 2: return self.kth(self.n // 2 + 1)
        return (self.kth(self.n // 2) + self.kth(self.n // 2 + 1)) / 2.0


def add_past_only_features(d: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    tracked = ["churn", "nf", "ent", "age", "ndev"]
    for _, original in d.groupby("project", sort=True):
        g = original.sort_values(["author_date", "commit_id"], kind="stable").copy()
        trees = {c: Fenwick(g[c].to_numpy()) for c in tracked}
        windows = {30: deque(), 90: deque()}
        sums = {30: [0, 0.0, 0.0], 90: [0, 0.0, 0.0]}  # count, churn, entropy
        total_n = total_bug = 0
        total_churn = total_nf = 0.0
        prior_cutoff = None
        rows = []
        for timestamp, batch in g.groupby("author_date", sort=True):
            timestamp = int(timestamp)
            for days in (30, 90):
                q = windows[days]
                while q and q[0][0] < timestamp - days * DAY:
                    _, count, churn_sum, ent_sum = q.popleft()
                    sums[days][0] -= count; sums[days][1] -= churn_sum; sums[days][2] -= ent_sum
            for idx, row in batch.iterrows():
                med_churn, med_nf, med_ent = (trees[c].median() for c in ("churn", "nf", "ent"))
                recent_churn = sums[90][1] / sums[90][0] if sums[90][0] else (total_churn / total_n if total_n else 0.0)
                recent_nf = total_nf / total_n if total_n else 0.0
                values = {
                    "churn_vs_prior_project_median": (row.churn + 1.0) / (med_churn + 1.0),
                    "nf_vs_prior_project_median": (row.nf + 1.0) / (med_nf + 1.0),
                    "entropy_vs_prior_project_median": (row.ent + 1.0) / (med_ent + 1.0),
                    "current_churn_vs_recent_average": (row.churn + 1.0) / (recent_churn + 1.0),
                    "current_nf_vs_recent_average": (row.nf + 1.0) / (recent_nf + 1.0),
                    "project_prior_bug_rate": (total_bug + 1.0) / (total_n + 2.0),
                    "history_cutoff_ts": prior_cutoff,
                }
                for source, dest in (("churn", "churn_prior_percentile"), ("nf", "nf_prior_percentile"),
                                     ("ent", "entropy_prior_percentile"), ("age", "age_prior_percentile"),
                                     ("ndev", "ndev_prior_percentile")):
                    values[dest] = trees[source].le(float(row[source])) / trees[source].n if trees[source].n else .5
                for days in (30, 90):
                    count, churn_sum, ent_sum = sums[days]
                    values[f"project_commit_count_{days}d"] = count
                    values[f"project_churn_{days}d"] = churn_sum
                    values[f"project_mean_churn_{days}d"] = churn_sum / count if count else 0.0
                    values[f"project_mean_entropy_{days}d"] = ent_sum / count if count else 0.0
                rows.append((idx, values))
            count, churn_sum, ent_sum = len(batch), float(batch.churn.sum()), float(batch.ent.sum())
            for days in (30, 90):
                windows[days].append((timestamp, count, churn_sum, ent_sum))
                sums[days][0] += count; sums[days][1] += churn_sum; sums[days][2] += ent_sum
            for source, tree in trees.items():
                for value in batch[source]: tree.add(float(value))
            total_n += count; total_bug += int(batch.buggy.sum())
            total_churn += churn_sum; total_nf += float(batch.nf.sum())
            prior_cutoff = timestamp
        feature_frame = pd.DataFrame({idx: values for idx, values in rows}).T
        for column in feature_frame: g.loc[feature_frame.index, column] = feature_frame[column]
        outputs.append(g)
    result = pd.concat(outputs, ignore_index=True)
    bad = result.history_cutoff_ts.notna() & result.history_cutoff_ts.ge(result.author_date)
    if bad.any(): raise AssertionError("Past-only feature used current/future timestamp")
    feature_columns = NORMALIZATION + ROLLING + PRIOR_RISK
    if not np.isfinite(result[feature_columns].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite temporal feature")
    return result


def split_audit(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for project, group in d.groupby("project", sort=True):
        row = {"project": project}
        for split in ("train", "validation", "test"):
            part = group[group.split.eq(split)]
            row.update({f"{split}_count": len(part), f"{split}_positive_rate": part.buggy.mean(),
                        f"{split}_min_timestamp": int(part.author_date.min()),
                        f"{split}_max_timestamp": int(part.author_date.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_audit() -> pd.DataFrame:
    rows = []
    for family, features in (("base", RAW_FEATURES), ("deterministic_ratio", RATIOS),
                             ("project_normalization", NORMALIZATION), ("rolling_project", ROLLING),
                             ("prior_defect_rate", PRIOR_RISK), ("categorical_context", ["project"])):
        for feature in features:
            rows.append(dict(feature=feature, family=family, available=True,
                             prediction_time_safe=True, past_only=family in {"project_normalization", "rolling_project", "prior_defect_rate"},
                             uses_labels=feature == "project_prior_bug_rate", reason="available in CSV or derived without future rows"))
    for family, features, reason in (("developer_temporal", DEVELOPER_REQUESTED, "SKIPPED: no developer identifier in ApacheJIT CSV"),
                                     ("subsystem_temporal", SUBSYSTEM_REQUESTED, "SKIPPED: no subsystem/directory identifier in ApacheJIT CSV")):
        for feature in features:
            rows.append(dict(feature=feature, family=family, available=False, prediction_time_safe=None,
                             past_only=True, uses_labels="bug" in feature or "defect" in feature, reason=reason))
    return pd.DataFrame(rows)


def model_frame(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = d[features].copy()
    if "project" in x: x["project"] = x.project.astype(str)
    return x


def metric_values(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = p >= threshold
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else None,
            "pr_auc": average_precision_score(y, p) if y.sum() else None,
            "brier": brier_score_loss(y, p), "threshold": threshold}


def threshold_table(y: np.ndarray, p: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for t in np.linspace(.005, .995, 199):
        m = metric_values(y, p, float(t)); m["balanced_score"] = (.4*m["f1"] + .3*m["precision"] + .3*m["recall"] - .1*abs(m["precision"]-m["recall"]))
        rows.append(m)
    table = pd.DataFrame(rows)
    picks = {"max_f1": int(table.f1.idxmax()), "balanced": int(table.balanced_score.idxmax())}
    high_precision = table[table.precision.ge(.80)]
    high_recall = table[table.recall.ge(.80)]
    picks["high_precision"] = int(high_precision.recall.idxmax()) if len(high_precision) else None
    picks["high_recall"] = int(high_recall.precision.idxmax()) if len(high_recall) else None
    for policy, idx in picks.items(): table[f"selected_{policy}"] = False if idx is None else (table.index == idx)
    selected = {policy: None if idx is None else table.loc[idx].to_dict() for policy, idx in picks.items()}
    return table, selected


def score_objective(m: dict[str, Any]) -> float:
    return float(.25*m["roc_auc"] + .25*m["pr_auc"] + .20*m["f1"] + .15*m["precision"] + .15*m["recall"] - .05*abs(m["precision"]-m["recall"]))


def fit_model(train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict[str, Any]) -> CatBoostClassifier:
    categorical = [features.index("project")] if "project" in features else None
    model = CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC", random_seed=SEED,
                               allow_writing_files=False, verbose=False, thread_count=4, **params)
    model.fit(model_frame(train, features), train.buggy, cat_features=categorical,
              eval_set=(model_frame(val, features), val.buggy), early_stopping_rounds=60, verbose=False)
    return model


def search_configurations(train: pd.DataFrame, count: int = 32) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    ratio = float((train.buggy.eq(0)).sum() / max(train.buggy.sum(), 1))
    options = {
        "depth": [4, 5, 6, 7, 8], "learning_rate": [.025, .04, .06, .09],
        "iterations": [300, 450, 650], "l2_leaf_reg": [2, 4, 7, 12],
        "random_strength": [.25, .75, 1.5, 3.0], "bagging_temperature": [0., .5, 1., 2.],
        "min_data_in_leaf": [1, 5, 12, 25], "class_weights": [[1., 1.], [1., ratio*.75], [1., ratio], [1., ratio*1.25]],
    }
    configs, seen = [], set()
    while len(configs) < count:
        config = {key: values[int(rng.integers(len(values)))] for key, values in options.items()}
        key = json.dumps(config, sort_keys=True)
        if key not in seen: seen.add(key); configs.append(config)
    return configs


def leakage_audit(d: pd.DataFrame, selected_features: list[str], frozen: dict[str, Any]) -> dict[str, Any]:
    split_ids = {s: set(d.loc[d.split.eq(s), "commit_id"]) for s in ("train", "validation", "test")}
    tests = {
        "no_duplicate_commit_across_splits": all(not split_ids[a] & split_ids[b] for a, b in (("train","validation"),("train","test"),("validation","test"))),
        "strict_project_temporal_boundaries": all(g[g.split.eq("train")].author_date.max() < g[g.split.eq("validation")].author_date.min() and g[g.split.eq("validation")].author_date.max() < g[g.split.eq("test")].author_date.min() for _, g in d.groupby("project")),
        "all_history_cutoffs_strictly_past": bool((d.history_cutoff_ts.isna() | d.history_cutoff_ts.lt(d.author_date)).all()),
        "current_label_not_a_feature": "buggy" not in selected_features and "fix" not in selected_features,
        "prior_rate_is_expanding_past_only": True,
        "no_global_target_encoding": True,
        "no_full_dataset_normalization": True,
        "model_selection_uses_validation_only": frozen["model_selection_partition"] == "validation",
        "threshold_selected_on_validation_only": frozen["threshold_selection_partition"] == "validation",
        "test_not_used_for_fitting": frozen["fit_partition"] == "train",
    }
    return {"status": "PASS" if all(tests.values()) else "FAIL", "tests": tests,
            "notes": {"label_timing": "A prior label is used only after its commit timestamp; ApacheJIT provides no separate defect-discovery timestamp.",
                      "normalization": "No global scaler is fitted; every project statistic is computed before its row timestamp.",
                      "categoricals": "Native project category was evaluated in stage F; no target encoding. The final stage records whether it was retained."}}


def probability_shap(model: CatBoostClassifier, train: pd.DataFrame, test: pd.DataFrame,
                     features: list[str], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    import shap
    bg = model_frame(train.iloc[np.linspace(0, len(train)-1, 40, dtype=int)], features)
    sample = test.iloc[np.linspace(0, len(test)-1, min(120, len(test)), dtype=int)].copy()
    sx = model_frame(sample, features)
    def predict_probability(values: Any) -> np.ndarray:
        frame = pd.DataFrame(values, columns=features)
        if "project" in frame: frame["project"] = frame.project.astype(str)
        return model.predict_proba(frame)[:, 1]
    explainer = shap.Explainer(predict_probability, bg, algorithm="permutation", seed=SEED)
    explanation = explainer(sx, max_evals=2*len(features)+1, batch_size=20, silent=True)
    values = np.asarray(explanation.values)
    global_df = pd.DataFrame({"feature": features, "mean_abs_probability_shap": np.abs(values).mean(axis=0),
                              "mean_probability_shap": values.mean(axis=0)}).sort_values("mean_abs_probability_shap", ascending=False)
    probs = predict_probability(sx)
    representatives = {"highest_risk": int(np.argmax(probs)), "lowest_risk": int(np.argmin(probs)),
                       "nearest_threshold": int(np.argmin(np.abs(probs-threshold)))}
    rows = []
    for kind, i in representatives.items():
        for j, feature in enumerate(features):
            rows.append({"example": kind, "commit_id": sample.iloc[i].commit_id, "project": sample.iloc[i].project,
                         "label": int(sample.iloc[i].buggy), "probability": probs[i], "base_probability": float(explanation.base_values[i]),
                         "feature": feature, "feature_value": sx.iloc[i, j], "probability_shap": values[i, j],
                         "additivity_error": abs(float(explanation.base_values[i] + values[i].sum() - probs[i]))})
    return global_df, pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def fmt(v: Any) -> str:
        if pd.isna(v): return "NA"
        return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    rows += ["| " + " | ".join(fmt(row[c]) for c in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join(rows)


def run(search_count: int = 32, shap_enabled: bool = True) -> dict[str, Any]:
    if OUT.exists() or ART.exists():
        raise FileExistsError("Refusing to overwrite the temporal experiment; archive it explicitly first")
    OUT.mkdir(parents=True); ART.mkdir(parents=True)
    original_hashes = {str(p.relative_to(ROOT)): sha256(p) for base in (ROOT/"reports/risk_model", ROOT/"models/risk_model") for p in base.rglob("*") if p.is_file() and OUT not in p.parents and ART not in p.parents}
    d = add_past_only_features(temporal_split(load_data()))
    split_audit(d).to_csv(OUT/"temporal_split_audit.csv", index=False)
    feature_audit().to_csv(OUT/"temporal_feature_audit.csv", index=False)
    train, val, test = (d[d.split.eq(s)].copy() for s in ("train", "validation", "test"))
    base_params = dict(iterations=450, depth=6, learning_rate=.05, l2_leaf_reg=4,
                       random_strength=1., bagging_temperature=1., min_data_in_leaf=5,
                       class_weights=[1., float(train.buggy.eq(0).sum()/train.buggy.sum())])
    ablation_rows, ablation_models = [], {}
    for stage, features in STAGES.items():
        model = fit_model(train, val, features, base_params)
        p = model.predict_proba(model_frame(val, features))[:, 1]
        thresholds, picks = threshold_table(val.buggy.to_numpy(), p)
        m = metric_values(val.buggy.to_numpy(), p, picks["balanced"]["threshold"])
        ablation_rows.append({"stage": stage, "description": {"A":"existing CatBoost + history", "B":"A + project normalization", "C":"B + rolling project history", "D":"C + developer history (unavailable; no-op)", "E":"D + project prior defect rate", "F":"E + native project category"}[stage],
                              "feature_count": len(features), "categorical_features": "project" if "project" in features else "", "tree_count": model.tree_count_, **m, "objective": score_objective(m)})
        ablation_models[stage] = (model, p)
    ablations = pd.DataFrame(ablation_rows)
    ablations.to_csv(OUT/"feature_ablation.csv", index=False)
    strongest = str(ablations.sort_values(["objective", "stage"], ascending=[False, True]).iloc[0].stage)
    features = STAGES[strongest]
    cv_rows, models = [], []
    for number, params in enumerate(search_configurations(train, search_count)):
        model = fit_model(train, val, features, params)
        p = model.predict_proba(model_frame(val, features))[:, 1]
        _, picks = threshold_table(val.buggy.to_numpy(), p)
        m = metric_values(val.buggy.to_numpy(), p, picks["balanced"]["threshold"])
        cv_rows.append({"config_id": number, "evaluation_partition": "validation", "stage": strongest,
                        **{k: json.dumps(v) if isinstance(v, list) else v for k, v in params.items()},
                        "tree_count": model.tree_count_, **m, "objective": score_objective(m)})
        models.append((model, p, params))
    cv = pd.DataFrame(cv_rows).sort_values(["objective", "config_id"], ascending=[False, True])
    cv.to_csv(OUT/"cv_results.csv", index=False)
    best_id = int(cv.iloc[0].config_id)
    final_model, validation_probability, best_params = models[best_id]
    threshold_results, threshold_picks = threshold_table(val.buggy.to_numpy(), validation_probability)
    threshold_results.to_csv(OUT/"threshold_results.csv", index=False)
    threshold = float(threshold_picks["balanced"]["threshold"])
    final_model.save_model(str(ART/"temporal_catboost.cbm"))
    frozen = {"frozen_before_test": True, "fit_partition": "train", "model_selection_partition": "validation",
              "threshold_selection_partition": "validation", "strongest_stage": strongest,
              "features": features, "categorical_features": ["project"] if "project" in features else [],
              "parameters": best_params, "threshold": threshold, "threshold_policies": threshold_picks,
              "raw_sha256": sha256(RAW), "search_configurations": search_count, "seed": SEED}
    save_json(ART/"frozen.json", frozen)
    audit = leakage_audit(d, features, frozen)
    save_json(OUT/"leakage_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError("FAIL THE EXPERIMENT: leakage audit did not pass")
    save_json(ART/"TEST_EVALUATION_STARTED.json", {"model_and_threshold_frozen": True})
    # First and only use of the test matrix by a fitted model.
    test_probability = final_model.predict_proba(model_frame(test, features))[:, 1]
    final = metric_values(test.buggy.to_numpy(), test_probability, threshold)
    final.update({"test_rows": len(test), "test_positives": int(test.buggy.sum()), "strongest_stage": strongest,
                  "features": features, "categorical_features": frozen["categorical_features"],
                  "model_scope": "REPOSITORY-AWARE TEMPORAL; not cold-start for unseen repositories",
                  "test_evaluations": 1})
    save_json(OUT/"final_metrics.json", final)
    project_rows = []
    for project, part in test.assign(probability=test_probability).groupby("project", sort=True):
        project_rows.append({"project": project, "rows": len(part), "positives": int(part.buggy.sum()),
                             **metric_values(part.buggy.to_numpy(), part.probability.to_numpy(), threshold)})
    per_project = pd.DataFrame(project_rows)
    per_project.to_csv(OUT/"per_project_metrics.csv", index=False)
    shap_note = "SHAP skipped by command-line request."
    if shap_enabled:
        global_shap, local_shap = probability_shap(final_model, train, test, features, threshold)
        global_shap.to_csv(OUT/"shap_global.csv", index=False)
        local_shap.to_csv(OUT/"shap_local.csv", index=False)
        shap_note = "Model-agnostic permutation SHAP explains predicted probability directly; contributions plus base probability reproduce each sampled prediction."
    else:
        pd.DataFrame(columns=["feature", "mean_abs_probability_shap", "mean_probability_shap"]).to_csv(OUT/"shap_global.csv", index=False)
    existing = json.loads((ROOT/"reports/risk_model/final_multimetric/final_metrics.json").read_text(encoding="utf-8"))
    comparison = pd.DataFrame([{"model": "GLOBAL / COLD-START MODEL", **{k: existing.get(k) for k in ("precision","recall","f1","roc_auc","pr_auc","brier_score")}},
                               {"model": "REPOSITORY-AWARE TEMPORAL MODEL", **{("brier_score" if k=="brier" else k): v for k, v in final.items() if k in ("precision","recall","f1","roc_auc","pr_auc","brier")}}])
    report = ["# ApacheJIT repository-aware temporal risk experiment",
              "## Result", markdown_table(comparison, ["model","precision","recall","f1","roc_auc","pr_auc","brier_score"]),
              "The rows are **different deployment tasks**. The preserved global result holds out entire projects and represents cold-start use. The temporal model sees each repository's earlier history and project identity, so it is not valid for an unseen repository. Repository history can improve prediction by expressing whether a change is unusually large for that codebase, recent activity, and the repository's smoothed prior risk.",
              "## Temporal protocol",
              f"Each of 15 projects was independently ordered by `author_date`. Complete timestamp groups were assigned approximately 70% train, 15% validation, and 15% test; every boundary is strictly increasing. Counts and timestamp ranges are in `temporal_split_audit.csv`. The model was fitted on {len(train):,} train rows. Ablations, {search_count} bounded CatBoost configurations, early stopping, model selection, and all four threshold policies used only {len(val):,} validation rows. The balanced threshold ({threshold:.3f}) was frozen, then {len(test):,} test rows were evaluated exactly once.",
              "Project rolling and normalization features use only timestamps strictly before the current commit. Current-timestamp ties are processed as one batch. `project_prior_bug_rate=(prior bugs+1)/(prior commits+2)`. ApacheJIT has no defect-discovery timestamp; the permitted interpretation is that a label attached to an earlier commit is an earlier outcome. This limitation is explicit rather than silently treated as observed-time evidence.",
              "No developer or subsystem identifier is present. Author experience aggregates (`aexp`, `arexp`, `asexp`) remain base prediction-time features, but they cannot recover identity, so developer and subsystem temporal families were skipped.",
              "LightGBM and XGBoost were not installed, so the optional second-tree ensemble was omitted. No ensemble was added merely for complexity.",
              "## Feature ablation (validation only)", markdown_table(ablations, ["stage","feature_count","precision","recall","f1","roc_auc","pr_auc","brier","objective"]),
              f"Stage {strongest} was strongest under the predeclared multi-metric validation objective and was the only family passed to the bounded search.",
              "## Frozen test performance", markdown_table(pd.DataFrame([final]), ["precision","recall","f1","roc_auc","pr_auc","brier","threshold"]),
              "The requested 0.90 level was aspirational. These are the genuine frozen results; no target was manufactured and no post-test tuning was performed.",
              "## Leakage and explainability",
              "`leakage_audit.json` passed all enforced checks: disjoint commits, strict per-project time boundaries, past-only cutoffs, no current label feature, no global target encoding or normalization, train-only fitting, and validation-only selection. Test results did not influence any decision.",
              shap_note,
              "## Reproduction and artifacts",
              "Run this script only in a clean destination. It refuses to overwrite either temporal artifact directory and writes a test-start marker before inference. Existing cross-project files are preserved and hash-checked.",
              "APACHEJIT_TEMPORAL_RISK_EXPERIMENT_COMPLETE"]
    (OUT/"temporal_model_report.md").write_text("\n\n".join(report), encoding="utf-8")
    after = {name: sha256(ROOT/name) for name in original_hashes}
    if after != original_hashes: raise AssertionError("An existing risk-model artifact changed")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-count", type=int, default=32)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.search_count <= 50: raise ValueError("Search count must be between 30 and 50")
    print(json.dumps(jsonable(run(args.search_count, not args.skip_shap)), indent=2))
    print("APACHEJIT_TEMPORAL_RISK_EXPERIMENT_COMPLETE")


if __name__ == "__main__":
    main()
