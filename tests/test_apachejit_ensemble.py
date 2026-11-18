"""Protocol and utility checks for the development-only ensemble."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_apachejit_ensemble_experiment as ensemble
import run_apachejit_temporal_experiment as temporal


def tiny_frame():
    rows = []
    for project in ("p1", "p2"):
        for i, ts in enumerate(range(1, 21)):
            row = {"commit_id": f"{project}-{i}", "project": project, "author_date": ts,
                   "buggy": i % 3 == 0, "fix": False, "year": 2020}
            row.update({name: float(i + 1) for name in temporal.RAW_FEATURES})
            rows.append(row)
    d = pd.DataFrame(rows)
    d["churn"] = d.la + d.ld
    for name in temporal.RATIOS: d[name] = 0.
    return d


def test_rolling_assignments_are_disjoint_and_ordered():
    d = tiny_frame()
    role = ensemble.rolling_assignments(d, .4, .5, .6)
    assert set(role.unique()) == {"base_train", "inner", "outer", "excluded"}
    for _, group in d.assign(role=role).groupby("project"):
        assert group[group.role.eq("base_train")].author_date.max() < group[group.role.eq("inner")].author_date.min()
        assert group[group.role.eq("inner")].author_date.max() < group[group.role.eq("outer")].author_date.min()


def test_gating_limits_and_direction():
    g = np.array([.2, .2]); t = np.array([.8, .8]); h = np.array([0., 1000.])
    p = ensemble.gated_probability(g, t, h, 10.)
    assert p[0] == .2
    assert .79 < p[1] < .8


def test_dominance_requires_all_metrics_and_one_strict_gain():
    a = pd.Series({k: .8 for k in ensemble.TARGETS})
    b = pd.Series({k: .8 for k in ensemble.TARGETS})
    assert not ensemble.dominates(a, b)
    a["f1"] = .81
    assert ensemble.dominates(a, b)
    a["recall"] = .79
    assert not ensemble.dominates(a, b)


def test_threshold_grid_contains_requested_policies():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .4, .6, .9])
    table, selected = ensemble.threshold_grid(y, p)
    assert len(table) == 199
    assert set(selected) == {"max_f1", "balanced_precision_recall", "best_joint_objective", "precision_ge_070", "recall_ge_070"}
