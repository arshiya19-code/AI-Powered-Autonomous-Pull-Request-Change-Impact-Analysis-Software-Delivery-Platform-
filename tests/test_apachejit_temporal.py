"""Leakage guards for the repository-aware temporal experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_apachejit_temporal_experiment as temporal


def tiny_frame():
    rows = []
    for project in ("p1", "p2"):
        for i, ts in enumerate([1, 2, 2, 3, 4, 5, 6, 7, 8, 9]):
            row = {"commit_id": f"{project}-{i}", "project": project, "author_date": ts,
                   "buggy": i % 2, "fix": False, "year": 2020}
            row.update({name: float(i + 1) for name in temporal.RAW_FEATURES})
            rows.append(row)
    d = pd.DataFrame(rows)
    d["churn"] = d.la + d.ld
    for name in temporal.RATIOS: d[name] = 0.0
    return d


def test_temporal_split_keeps_timestamp_ties_together_and_is_strict():
    d = temporal.temporal_split(tiny_frame())
    for _, group in d.groupby("project"):
        assert group.groupby("author_date").split.nunique().max() == 1
        assert group[group.split.eq("train")].author_date.max() < group[group.split.eq("validation")].author_date.min()
        assert group[group.split.eq("validation")].author_date.max() < group[group.split.eq("test")].author_date.min()
    assert d.commit_id.is_unique


def test_current_timestamp_batch_does_not_enter_history_or_label_rate():
    d = temporal.add_past_only_features(temporal.temporal_split(tiny_frame()))
    tied = d[(d.project.eq("p1")) & d.author_date.eq(2)].sort_values("commit_id")
    # Both tied rows see only timestamp 1 and therefore identical history state.
    assert tied.project_prior_bug_rate.nunique() == 1
    assert tied.project_commit_count_30d.nunique() == 1
    assert (tied.history_cutoff_ts == 1).all()
    assert (d.history_cutoff_ts.dropna() < d.loc[d.history_cutoff_ts.notna(), "author_date"]).all()


def test_feature_stage_never_contains_target_or_posthoc_fix():
    for features in temporal.STAGES.values():
        assert "buggy" not in features
        assert "fix" not in features
    audit = temporal.feature_audit().set_index("feature")
    assert not audit.loc[temporal.DEVELOPER_REQUESTED, "available"].any()
    assert not audit.loc[temporal.SUBSYSTEM_REQUESTED, "available"].any()


def test_threshold_is_selected_without_test_input():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([.1, .8, .4, .7, .6, .2])
    table, selected = temporal.threshold_table(y, p)
    assert len(table) == 199
    assert selected["balanced"]["threshold"] == pytest.approx(table.loc[table.balanced_score.idxmax(), "threshold"])


def test_fenwick_exact_prior_percentile_and_median():
    tree = temporal.Fenwick(np.array([1., 2., 3., 4.]))
    tree.add(1.); tree.add(4.); tree.add(2.)
    assert tree.le(2.) == 2
    assert tree.median() == 2.
