"""Numerical checks for the independent ApacheJIT experiment."""
import numpy as np
import pandas as pd
import pytest

from scripts.compare_apachejit_models import audit, bootstrap, gating


def test_known_capture_and_nominal_lift():
    y = np.array([1, 0, 1, 0, 1] + [0] * 15)
    result = gating(y, np.arange(20, 0, -1))
    assert result['5']['gated_count'] == 1
    assert result['5']['capture'] == pytest.approx(1 / 3)
    assert result['5']['lift'] == pytest.approx((1 / 3) / .05)
    assert result['10']['captured_count'] == 1
    assert result['50']['capture'] == 1


def test_rounding_ties_and_no_positives():
    # Exact ties honor input order, never favor positive labels.
    g = gating([0, 1, 1], [.5, .5, .5])
    assert g['5']['gated_count'] == 1
    assert g['5']['capture'] == 0
    assert g['50']['gated_count'] == 2
    assert g['50']['capture'] == .5
    assert g['50']['actual_gating_fraction'] == pytest.approx(2 / 3)
    assert gating([0, 0], [.9, .1])['5']['capture'] is None


def test_paired_bootstrap_identical_predictions():
    y = np.tile([0, 1], 20)
    p = np.linspace(0, 1, len(y))
    result = bootstrap(y, {'Logistic Regression': p, 'CatBoost V1': p.copy()}, 100)
    for ci in result['differences_vs_logistic']['CatBoost V1'].values():
        assert ci['low'] == ci['high'] == 0
    for ci in result['capture_ratios']['CatBoost V1'].values():
        assert ci['low'] == ci['high'] == 1


def test_audit_excludes_outcome_and_identity():
    raw = pd.DataFrame({c: [1] for c in ['buggy', 'fix', 'project', 'commit_id', 'year', 'author_date', 'nf', 'ndev', 'aexp']})
    result = audit(raw).set_index('column')
    for col in ['buggy', 'fix', 'project', 'commit_id', 'year', 'author_date']:
        assert not result.loc[col, 'selected_v2']
    assert not result.loc['ndev', 'runtime_compatible']
    assert 'REQUIRES_HISTORY' in result.loc['aexp', 'classification']
