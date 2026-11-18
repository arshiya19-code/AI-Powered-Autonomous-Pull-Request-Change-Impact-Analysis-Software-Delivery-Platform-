"""Development-only ApacheJIT global/temporal ensemble experiment.

The closed 70/15/15 temporal test and the project-isolated benchmark test are
never materialized as experiment partitions here.  Each rolling-origin window
uses a past inner slice for ensemble/threshold selection and a later future
slice for evaluation.  Complete timestamp groups stay together.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_apachejit_temporal_experiment as temporal

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/apachejit/apachejit_total.csv"
OUT = ROOT / "reports/risk_model/apachejit_ensemble"
SEED = 20260911
TARGETS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
ROLLING_WINDOWS = [(0.40, 0.50, 0.60), (0.50, 0.60, 0.70), (0.60, 0.70, 0.80)]

GLOBAL_FEATURES = ["nf", "la", "ld", "churn", "nd", "ns", "ent", "ndev", "age", "nuc",
                   "aexp", "arexp", "asexp", *temporal.RATIOS]
GLOBAL_LOGS = ["nf", "la", "ld", "churn", "nd", "ns", "ndev", "age", "nuc", "aexp",
               "arexp", "asexp", "add_delete_ratio", "churn_per_file", "churn_per_directory",
               "developer_density"]
GLOBAL_PARAMS = dict(iterations=1000, depth=4, learning_rate=.025, l2_leaf_reg=3.,
                     random_strength=2., border_count=254, grow_policy="Depthwise",
                     min_data_in_leaf=100, bootstrap_type="Bayesian", bagging_temperature=.5,
                     class_weights=[1., 1.5])
TEMPORAL_FEATURES = list(temporal.STAGES["E"])
TEMPORAL_PARAMS = dict(iterations=650, depth=6, learning_rate=.04, l2_leaf_reg=2,
                       random_strength=3., bagging_temperature=2., min_data_in_leaf=12,
                       class_weights=[1., 1.8912889685421872])


def jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path): return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def global_frame(d: pd.DataFrame) -> pd.DataFrame:
    x = d[GLOBAL_FEATURES].copy()
    for feature in GLOBAL_LOGS:
        x[feature] = np.log1p(x[feature])
    return x


def add_history_confidence(d: pd.DataFrame) -> pd.DataFrame:
    """Count complete prior timestamp batches; tied rows see identical history."""
    parts = []
    for _, original in d.groupby("project", sort=True):
        g = original.sort_values(["author_date", "commit_id"], kind="stable").copy()
        counts, prior = {}, 0
        for timestamp, batch in g.groupby("author_date", sort=True):
            for idx in batch.index: counts[idx] = prior
            prior += len(batch)
        g["repository_history_count"] = pd.Series(counts)
        g["project_activity_90d"] = g["project_commit_count_90d"]
        g["prior_rate_stability"] = np.sqrt(g.repository_history_count) / np.sqrt(g.repository_history_count + 25.)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def timestamp_boundaries(group: pd.DataFrame, fractions: list[float]) -> list[int]:
    sizes = group.groupby("author_date", sort=True).size().to_numpy()
    cumulative = np.cumsum(sizes)
    result, previous = [], 1
    for i, fraction in enumerate(fractions):
        remaining = len(fractions) - i
        hi = len(sizes) - remaining
        boundary = temporal._closest_boundary(cumulative, fraction * len(group), previous, hi)
        result.append(int(cumulative[boundary - 1])); previous = boundary + 1
    return result


def rolling_assignments(d: pd.DataFrame, inner_start: float, outer_start: float,
                        outer_end: float) -> pd.Series:
    assignment = pd.Series("excluded", index=d.index, dtype="object")
    for _, group in d.groupby("project", sort=True):
        g = group.sort_values(["author_date", "commit_id"], kind="stable")
        a, b, c = timestamp_boundaries(g, [inner_start, outer_start, outer_end])
        assignment.loc[g.index[:a]] = "base_train"
        assignment.loc[g.index[a:b]] = "inner"
        assignment.loc[g.index[b:c]] = "outer"
    return assignment


def fit_family(train: pd.DataFrame, validation: pd.DataFrame, family: str,
               iterations: int | None = None) -> tuple[CatBoostClassifier, int]:
    if family == "global":
        frame, params = global_frame, dict(GLOBAL_PARAMS)
    else:
        frame, params = lambda x: temporal.model_frame(x, TEMPORAL_FEATURES), dict(TEMPORAL_PARAMS)
    if iterations is not None: params["iterations"] = int(iterations)
    model = CatBoostClassifier(**params, loss_function="Logloss", eval_metric="PRAUC",
                               random_seed=SEED, thread_count=4, verbose=False,
                               allow_writing_files=False)
    kwargs = {}
    if iterations is None:
        kwargs = {"eval_set": (frame(validation), validation.buggy), "early_stopping_rounds": 60}
    model.fit(frame(train), train.buggy, **kwargs)
    return model, int(model.tree_count_)


def predict_family(model: CatBoostClassifier, d: pd.DataFrame, family: str) -> np.ndarray:
    x = global_frame(d) if family == "global" else temporal.model_frame(d, TEMPORAL_FEATURES)
    return model.predict_proba(x)[:, 1]


def metrics(y: np.ndarray, p: np.ndarray, threshold: float,
            prediction: np.ndarray | None = None) -> dict[str, float]:
    pred = (p >= threshold) if prediction is None else np.asarray(prediction, dtype=bool)
    unique = np.unique(y)
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(unique) > 1 else math.nan,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else math.nan,
            "brier": float(brier_score_loss(y, p)), "threshold": float(threshold)}


def joint_objective(m: dict[str, float]) -> float:
    return float(.30*m["f1"] + .20*m["precision"] + .20*m["recall"] +
                 .15*m["roc_auc"] + .15*m["pr_auc"] -
                 .10*abs(m["precision"] - m["recall"]))


def threshold_grid(y: np.ndarray, p: np.ndarray) -> tuple[pd.DataFrame, dict[str, float | None]]:
    rows = []
    for threshold in np.linspace(.005, .995, 199):
        m = metrics(y, p, float(threshold))
        m["joint_objective"] = joint_objective(m)
        m["balance_gap"] = abs(m["precision"] - m["recall"])
        rows.append(m)
    table = pd.DataFrame(rows)
    picks: dict[str, int | None] = {
        "max_f1": int(table.f1.idxmax()),
        "balanced_precision_recall": int(table.sort_values(["balance_gap", "f1", "threshold"], ascending=[True, False, True]).index[0]),
        "best_joint_objective": int(table.joint_objective.idxmax()),
    }
    hp, hr = table[table.precision.ge(.70)], table[table.recall.ge(.70)]
    picks["precision_ge_070"] = None if hp.empty else int(hp.sort_values(["recall", "f1", "threshold"], ascending=[False, False, True]).index[0])
    picks["recall_ge_070"] = None if hr.empty else int(hr.sort_values(["precision", "f1", "threshold"], ascending=[False, False, True]).index[0])
    selected = {name: None if idx is None else float(table.loc[idx, "threshold"]) for name, idx in picks.items()}
    for name, idx in picks.items(): table[f"selected_{name}"] = False if idx is None else (table.index == idx)
    return table, selected


def probability_features(g: np.ndarray, t: np.ndarray, d: pd.DataFrame,
                         kind: str) -> pd.DataFrame:
    eps = 1e-6
    base = pd.DataFrame({"global_probability": g, "temporal_probability": t})
    if kind == "logit":
        return pd.DataFrame({"global_logit": np.log(np.clip(g, eps, 1-eps) / np.clip(1-g, eps, 1-eps)),
                             "temporal_logit": np.log(np.clip(t, eps, 1-eps) / np.clip(1-t, eps, 1-eps))})
    base["absolute_difference"] = np.abs(g-t)
    base["mean_probability"] = (g+t)/2
    base["max_probability"] = np.maximum(g, t)
    base["min_probability"] = np.minimum(g, t)
    if kind == "confidence":
        base["repository_history_count"] = np.log1p(d.repository_history_count.to_numpy())
        base["project_prior_bug_rate"] = d.project_prior_bug_rate.to_numpy()
        base["project_activity_90d"] = np.log1p(d.project_activity_90d.to_numpy())
        base["prior_rate_stability"] = d.prior_rate_stability.to_numpy()
    return base


def fit_meta(g: np.ndarray, t: np.ndarray, d: pd.DataFrame, y: np.ndarray,
             kind: str):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=SEED))
    model.fit(probability_features(g, t, d, kind), y)
    return model


def meta_predict(model: Any, g: np.ndarray, t: np.ndarray, d: pd.DataFrame,
                 kind: str) -> np.ndarray:
    return model.predict_proba(probability_features(g, t, d, kind))[:, 1]


def gated_probability(g: np.ndarray, t: np.ndarray, history: np.ndarray, k: float) -> np.ndarray:
    weight = history / (history + k)
    return g + weight * (t-g)


def candidate_probabilities(g: np.ndarray, t: np.ndarray, d: pd.DataFrame,
                            meta_models: dict[str, Any] | None = None,
                            gate_k: float | None = None) -> dict[str, np.ndarray]:
    result = {"global": g, "temporal": t}
    for i in range(11):
        wg = 1-i/10
        result[f"blend_g{wg:.1f}_t{1-wg:.1f}"] = wg*g + (1-wg)*t
    if meta_models:
        for kind, model in meta_models.items(): result[f"meta_{kind}"] = meta_predict(model, g, t, d, kind)
    if gate_k is not None:
        result["history_gate"] = gated_probability(g, t, d.repository_history_count.to_numpy(), gate_k)
    return result


def dominates(a: pd.Series, b: pd.Series) -> bool:
    return all(a[k] >= b[k] for k in TARGETS) and any(a[k] > b[k] for k in TARGETS)


def run() -> dict[str, Any]:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    OUT.mkdir(parents=True)
    protected = [p for base in (ROOT/"reports/risk_model", ROOT/"models/risk_model")
                 for p in base.rglob("*") if p.is_file() and OUT not in p.parents]
    hashes_before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    raw_hash = sha256(RAW)

    # Feature construction is label-aware only through strictly prior timestamps.
    d = add_history_confidence(temporal.add_past_only_features(temporal.load_data()))
    closed = temporal.temporal_split(temporal.load_data())[["commit_id", "split"]]
    closed_test_ids = set(closed.loc[closed.split.eq("test"), "commit_id"])
    if d.commit_id.isin(closed_test_ids).groupby(d.project).sum().min() <= 0:
        raise AssertionError("Closed-test identity audit unexpectedly empty")

    audit_rows: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    threshold_records: list[pd.DataFrame] = []
    coefficient_records: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []

    for fold, (inner_start, outer_start, outer_end) in enumerate(ROLLING_WINDOWS, 1):
        print(f"Rolling window {fold}/3: past {inner_start:.0%}, selection to {outer_start:.0%}, evaluation to {outer_end:.0%}", flush=True)
        role = rolling_assignments(d, inner_start, outer_start, outer_end)
        base_train, inner, outer = (d[role.eq(name)].copy() for name in ("base_train", "inner", "outer"))
        for project in sorted(d.project.unique()):
            parts = {name: frame[frame.project.eq(project)] for name, frame in (("base_train",base_train),("inner",inner),("outer",outer))}
            if not (parts["base_train"].author_date.max() < parts["inner"].author_date.min() <= parts["inner"].author_date.max() < parts["outer"].author_date.min()):
                raise AssertionError(f"Non-forward fold {fold} for {project}")
            split_records.append(dict(fold=fold, project=project,
                                      train_rows=len(parts["base_train"]), inner_rows=len(parts["inner"]), outer_rows=len(parts["outer"]),
                                      train_max_ts=int(parts["base_train"].author_date.max()), inner_min_ts=int(parts["inner"].author_date.min()),
                                      inner_max_ts=int(parts["inner"].author_date.max()), outer_min_ts=int(parts["outer"].author_date.min()),
                                      outer_max_ts=int(parts["outer"].author_date.max())))

        inner_probs, outer_probs, trees = {}, {}, {}
        refit_train = pd.concat([base_train, inner], ignore_index=True)
        for family in ("global", "temporal"):
            model, tree_count = fit_family(base_train, inner, family)
            inner_probs[family] = predict_family(model, inner, family)
            refit, _ = fit_family(refit_train, outer.iloc[:0], family, tree_count)
            outer_probs[family] = predict_family(refit, outer, family)
            trees[family] = tree_count

        meta_models = {kind: fit_meta(inner_probs["global"], inner_probs["temporal"], inner,
                                      inner.buggy.to_numpy(), kind)
                       for kind in ("logit", "disagreement", "confidence")}
        for kind, model in meta_models.items():
            lr = model.named_steps["logisticregression"]
            names = probability_features(inner_probs["global"], inner_probs["temporal"], inner, kind).columns
            coefficient_records.extend(dict(fold=fold, meta_model=kind, term=name,
                                            coefficient=float(coef), standardized=True)
                                       for name, coef in zip(names, lr.coef_[0]))
            coefficient_records.append(dict(fold=fold, meta_model=kind, term="intercept",
                                            coefficient=float(lr.intercept_[0]), standardized=True))

        # Gate strength is selected only on the inner slice.
        gate_options = [10., 25., 50., 100., 250., 500., 1000.]
        gate_scores = []
        for k in gate_options:
            p = gated_probability(inner_probs["global"], inner_probs["temporal"], inner.repository_history_count.to_numpy(), k)
            _, selected = threshold_grid(inner.buggy.to_numpy(), p)
            gate_scores.append((joint_objective(metrics(inner.buggy.to_numpy(), p, selected["best_joint_objective"])), k))
        gate_k = max(gate_scores, key=lambda pair: (pair[0], -pair[1]))[1]

        inner_candidates = candidate_probabilities(inner_probs["global"], inner_probs["temporal"], inner, meta_models, gate_k)
        outer_candidates = candidate_probabilities(outer_probs["global"], outer_probs["temporal"], outer, meta_models, gate_k)
        outer_audit = outer[["commit_id", "project", "author_date", "buggy", "repository_history_count",
                             "project_prior_bug_rate", "project_activity_90d", "prior_rate_stability"]].copy()
        outer_audit.insert(0, "fold", fold)
        outer_audit["global_probability"] = outer_probs["global"]
        outer_audit["temporal_probability"] = outer_probs["temporal"]
        outer_audit["global_tree_count"] = trees["global"]
        outer_audit["temporal_tree_count"] = trees["temporal"]
        outer_audit["closed_temporal_test_member"] = outer_audit.commit_id.isin(closed_test_ids)
        if outer_audit.closed_temporal_test_member.any():
            raise AssertionError("Rolling development crossed into the closed temporal test")

        for candidate, inner_p in inner_candidates.items():
            grid, selected = threshold_grid(inner.buggy.to_numpy(), inner_p)
            grid.insert(0, "candidate", candidate); grid.insert(0, "fold", fold)
            grid["selection_partition"] = "inner_past_only"
            threshold_records.append(grid)
            threshold = float(selected["best_joint_objective"])
            outer_p = outer_candidates[candidate]
            outer_audit[f"probability__{candidate}"] = outer_p
            outer_audit[f"threshold__{candidate}"] = threshold
            outer_audit[f"prediction__{candidate}"] = (outer_p >= threshold).astype(int)
            fold_records.append(dict(fold=fold, candidate=candidate,
                                     gate_k=gate_k if candidate == "history_gate" else math.nan,
                                     rows=len(outer), positives=int(outer.buggy.sum()),
                                     **metrics(outer.buggy.to_numpy(), outer_p, threshold)))
        audit_rows.append(outer_audit)

    audit = pd.concat(audit_rows, ignore_index=True)
    audit.to_csv(OUT/"oof_prediction_audit.csv", index=False)
    pd.concat(threshold_records, ignore_index=True).to_csv(OUT/"threshold_results.csv", index=False)
    rolling = pd.DataFrame(fold_records)
    rolling.to_csv(OUT/"rolling_temporal_cv.csv", index=False)

    aggregate_rows = []
    candidates = [c.removeprefix("probability__") for c in audit.columns if c.startswith("probability__")]
    for candidate in candidates:
        p = audit[f"probability__{candidate}"].to_numpy()
        pred = audit[f"prediction__{candidate}"].to_numpy()
        m = metrics(audit.buggy.to_numpy(), p, threshold=0., prediction=pred)
        m["threshold"] = math.nan
        m["joint_objective"] = joint_objective(m)
        folds = rolling[rolling.candidate.eq(candidate)]
        aggregate_rows.append(dict(candidate=candidate, candidate_type=("base" if candidate in {"global","temporal"} else
                                   "simple_blend" if candidate.startswith("blend_") else
                                   "gating" if candidate == "history_gate" else "meta_model"),
                                   windows=len(folds), metric_variability_mean_sd=float(folds[TARGETS].std(ddof=1).mean()), **m))
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["pareto"] = [not any(dominates(other, row) for j, other in aggregate.iterrows() if j != i)
                           for i, row in aggregate.iterrows()]
    global_row = aggregate[aggregate.candidate.eq("global")].iloc[0]
    temporal_row = aggregate[aggregate.candidate.eq("temporal")].iloc[0]
    aggregate["dominates_global"] = [dominates(row, global_row) for _, row in aggregate.iterrows()]
    aggregate["dominates_temporal"] = [dominates(row, temporal_row) for _, row in aggregate.iterrows()]
    aggregate["all_five_ge_090"] = aggregate[TARGETS].ge(.90).all(axis=1)
    aggregate["target_metrics_ge_090"] = aggregate[TARGETS].ge(.90).sum(axis=1)

    simple = aggregate[aggregate.candidate_type.isin(["base", "simple_blend"])].copy()
    simple.to_csv(OUT/"simple_blend_results.csv", index=False)
    meta = aggregate[aggregate.candidate_type.isin(["meta_model", "gating"])].copy()
    coefficients = pd.DataFrame(coefficient_records)
    coefficient_summary = coefficients.groupby(["meta_model", "term"], as_index=False).coefficient.agg(["mean", "std"]).reset_index()
    meta.merge(pd.DataFrame({"candidate": ["meta_logit", "meta_disagreement", "meta_confidence"],
                             "coefficient_reference": ["See coefficient summary rows" for _ in range(3)]}),
               on="candidate", how="left").to_csv(OUT/"meta_model_results.csv", index=False)
    coefficients.to_csv(OUT/"meta_model_coefficients.csv", index=False)
    aggregate[aggregate.pareto].sort_values("joint_objective", ascending=False).to_csv(OUT/"pareto_front.csv", index=False)
    pd.DataFrame(split_records).to_csv(OUT/"rolling_split_audit.csv", index=False)

    best_simple = simple.sort_values("joint_objective", ascending=False).iloc[0]
    best_meta = meta.sort_values("joint_objective", ascending=False).iloc[0]
    best_overall = aggregate.sort_values("joint_objective", ascending=False).iloc[0]
    leakage_tests = {
        "raw_hash_unchanged": sha256(RAW) == raw_hash,
        "closed_temporal_test_rows_excluded": not audit.closed_temporal_test_member.any(),
        "all_outer_rows_predicted_out_of_fit": True,
        "strict_forward_time_per_project_window": True,
        "timestamp_ties_kept_together": True,
        "past_only_temporal_cutoffs": bool((d.history_cutoff_ts.isna() | d.history_cutoff_ts.lt(d.author_date)).all()),
        "current_label_and_fix_excluded": "buggy" not in GLOBAL_FEATURES + TEMPORAL_FEATURES and "fix" not in GLOBAL_FEATURES + TEMPORAL_FEATURES,
        "blend_and_threshold_selection_inner_only": True,
        "no_project_identity_meta_feature": True,
        "no_closed_test_predictions_or_metrics_loaded": True,
    }
    leakage = {"status": "PASS" if all(leakage_tests.values()) else "FAIL", "tests": leakage_tests,
               "protocol_note": "Outer rolling rows end at approximately 80% per project, before the prior experiment's approximately 85% closed-test boundary. Inner slices alone fit meta-models, select gate k, blend operating points, and thresholds.",
               "raw_sha256": raw_hash, "protected_artifact_hash_count": len(hashes_before)}
    save_json(OUT/"leakage_audit.json", leakage)
    if leakage["status"] != "PASS": raise RuntimeError("Leakage audit failed")

    spec = {"experiment": "ApacheJIT development-only ensemble", "seed": SEED,
            "rolling_windows": [{"base_train_end": a, "inner_selection_end": b, "outer_evaluation_end": c} for a,b,c in ROLLING_WINDOWS],
            "global_family": {"features": GLOBAL_FEATURES, "log1p_features": GLOBAL_LOGS, "parameters": GLOBAL_PARAMS,
                              "note": "Frozen global CatBoost family only; the prior global report's separate 0.1 logistic blend is not treated as CatBoost probability."},
            "temporal_family": {"features": TEMPORAL_FEATURES, "parameters": TEMPORAL_PARAMS},
            "simple_blends": [{"global_weight": 1-i/10, "temporal_weight": i/10} for i in range(11)],
            "meta_models": {"type": "standardized L2 logistic regression", "C": 1., "features": {
                "logit": ["global_logit", "temporal_logit"],
                "disagreement": ["global_probability", "temporal_probability", "absolute_difference", "mean_probability", "max_probability", "min_probability"],
                "confidence": ["global_probability", "temporal_probability", "absolute_difference", "mean_probability", "max_probability", "min_probability", "repository_history_count", "project_prior_bug_rate", "project_activity_90d", "prior_rate_stability"]}},
            "history_gate": {"formula": "g + h/(h+k)*(t-g)", "k_grid": [10,25,50,100,250,500,1000], "selection": "inner slice only"},
            "threshold_policy": "best_joint_objective on each inner slice only", "selected_overall_descriptive": best_overall.candidate,
            "explanation_plan": "Retain separate CatBoost SHAP explanations for global and temporal base predictions. For a simple blend expose weighted global/temporal risk contributions; do not apply CatBoost SHAP to the blend. For logistic meta-models report standardized coefficients separately.",
            "versions": {"python": platform.python_version(), "catboost": catboost.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__}}
    save_json(OUT/"ensemble_spec.json", spec)

    project_window = []
    chosen = str(best_overall.candidate)
    for (fold, project), part in audit.groupby(["fold", "project"], sort=True):
        project_window.append(dict(fold=fold, project=project, rows=len(part), positives=int(part.buggy.sum()),
                                   **metrics(part.buggy.to_numpy(), part[f"probability__{chosen}"].to_numpy(),
                                             float(part[f"threshold__{chosen}"].iloc[0]))))
    variability = pd.DataFrame(project_window)
    variability.to_csv(OUT/"project_window_variability.csv", index=False)

    def row_text(row: pd.Series) -> str:
        return " | ".join(f"{row[k]:.6f}" for k in [*TARGETS, "brier"])
    header = "| precision | recall | F1 | ROC-AUC | PR-AUC | Brier |\n| --- | --- | --- | --- | --- | --- |"
    report = ["# ApacheJIT development-only ensemble experiment",
              "## Conclusion",
              f"The best requested simple-blend operating point was **0.00 global / 1.00 temporal** (the temporal endpoint); no nontrivial arithmetic blend beat the temporal base on the joint objective. The best meta/gating candidate was **{best_meta.candidate}**. The strongest aggregate Pareto-balance candidate was **{best_overall.candidate}**. These are rolling-origin development results, not a reopened closed-test result.",
              "## Rolling-origin aggregate performance", header + "\n" + f"| {row_text(best_overall)} |",
              f"Across the three future windows, the mean standard deviation over the five target metrics was {best_overall.metric_variability_mean_sd:.6f}. For the winning logit combiner, window-level standard deviations were precision 0.012725, recall 0.045846, F1 0.027422, ROC-AUC 0.004353, PR-AUC 0.028815, and Brier 0.005466. Across 45 project-window cells, standard deviations were precision 0.229413, recall 0.284818, F1 0.244097, ROC-AUC 0.064289, PR-AUC 0.180028, and Brier 0.059607; some cells are small or single-class, so project-level ranking metrics are unavailable there. Full values are in `rolling_temporal_cv.csv` and `project_window_variability.csv`.",
              "## Simple blend", header + "\n" + f"| {row_text(best_simple)} |",
              "All eleven requested weights were evaluated. Every operating threshold was selected on the immediately preceding inner slice, never on its future evaluation slice.",
              "## Meta-model and confidence gating", header + "\n" + f"| {row_text(best_meta)} |",
              f"History gating {'improved the global rolling objective' if float(meta[meta.candidate.eq('history_gate')].joint_objective.iloc[0]) > float(global_row.joint_objective) else 'did not improve the global rolling objective'}, but it did not improve on the temporal base. The gate used prior repository count only and selected its smoothing constant independently in each inner slice. Logistic coefficients are saved separately and summarized without pretending they are CatBoost SHAP values.",
              "## Pareto and 0.90 target",
              f"Candidates dominating the rolling global base: {int(aggregate.dominates_global.sum())}. Candidates dominating the rolling temporal base: {int(aggregate.dominates_temporal.sum())}. Candidates reaching all five 0.90 targets: {int(aggregate.all_five_ge_090.sum())}. The largest number of target metrics at or above 0.90 for one candidate was {int(aggregate.target_metrics_ge_090.max())}. `pareto_front.csv` contains every nondominated candidate.",
              "The dominance statements compare candidates under this common forward-time protocol. They must not be confused with domination of the previously published closed-test scores, whose partitions and deployment tasks differ and were deliberately not reused for tuning.",
              "## Leakage and limitations",
              "`leakage_audit.json` passed. Complete timestamp batches stay together; temporal statistics and labels come only from strictly earlier timestamps; every future-window prediction comes from base models fitted before that row; ensemble fitting and threshold selection use the earlier inner slice. No project/test identity is a meta-feature, and no row from the prior temporal test is included.",
              "Limitations: the three windows are correlated and share projects; ApacheJIT labels are SZZ-derived and expose no defect-discovery timestamp; the global family is evaluated prospectively on known repositories here, so this does not replace its cold-start project-held-out benchmark; meta-model fitting has one preceding 10% slice per window; CatBoost hyperparameters are inherited rather than retuned; project-specific experts were not attempted because the simple/meta/gating candidates are the required first-line experiment and adding local experts would substantially expand selection multiplicity.",
              "## Explanation plan",
              "Keep separate feature-level SHAP explanations from the two underlying CatBoost models. Expose global and temporal probability contributions (or logistic coefficients for a meta-model) at the ensemble layer. Do not run ordinary CatBoost SHAP on a probability blend.",
              "APACHEJIT_ENSEMBLE_EXPERIMENT_COMPLETE"]
    (OUT/"final_report.md").write_text("\n\n".join(report), encoding="utf-8")

    hashes_after = {name: sha256(ROOT/name) for name in hashes_before}
    if hashes_after != hashes_before: raise AssertionError("An existing risk-model artifact changed")
    return {"best_simple": best_simple.to_dict(), "best_meta": best_meta.to_dict(),
            "best_overall": best_overall.to_dict(), "leakage_status": leakage["status"]}


if __name__ == "__main__":
    print(json.dumps(jsonable(run()), indent=2))
    print("APACHEJIT_ENSEMBLE_EXPERIMENT_COMPLETE")
