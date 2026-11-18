"""Numerical guards for the final research phase; no benchmark data required."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import optimize_apachejit_multimetric as opt


def test_threshold_grid_matches_independent_brute_force_with_ties():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0, .2, .2, .5, .5, .8, 1, 1])
    grid = np.linspace(.005, .995, 199)
    scores = []
    for t in grid:
        pp = precision_score(y, p >= t, zero_division=0)
        rr = recall_score(y, p >= t, zero_division=0)
        ff = f1_score(y, p >= t, zero_division=0)
        scores.append(.4*ff + .3*pp + .3*rr - .1*abs(pp-rr))
    selected = opt.thresholds(y, p)['balanced']
    assert selected['threshold'] == grid[np.argmax(scores)]
    assert selected['balanced_score'] == pytest.approx(max(scores))


def test_unattainable_high_precision_is_explicit():
    result = opt.thresholds(np.array([0,0,1]), np.array([.9,.8,.1]))
    assert result['high_precision'] is None
    assert result['high_recall']['recall'] >= .8


def test_derived_zero_denominators_and_no_label_metadata_inputs():
    d = pd.DataFrame({k: [0., 2.] for k in opt.EXPANDED})
    d['buggy'] = [0,1]; d['fix']=[True,False]; d['project']=['a','b']
    x = opt.derive(d)
    assert np.isfinite(x).all().all()
    assert set(x) == set(opt.EXPANDED + opt.DERIVED)
    altered = d.assign(buggy=[1,0], fix=[False,True], project=['z','y'])
    pd.testing.assert_frame_equal(x, opt.derive(altered))


def test_log_and_raw_matrix_and_pruning_order():
    d = pd.DataFrame({k: [0., 3.] for k in opt.EXPANDED})
    spec = dict(source_features=opt.BASE, transform='both', logs=[opt.BASE[0]])
    x = opt.matrix(d, spec)
    assert x.iloc[1][opt.BASE[0]] == 3
    assert x.iloc[1]['log1p_'+opt.BASE[0]] == pytest.approx(np.log(4))
    keep = ['log1p_'+opt.BASE[0],opt.BASE[1]]
    assert opt.matrix(d,dict(spec,keep=keep)).columns.tolist() == keep


def test_objective_penalizes_precision_recall_imbalance():
    common = dict(roc_auc=.8,pr_auc=.7,f1=.6)
    assert opt.objective(dict(common,precision=.6,recall=.6)) > opt.objective(dict(common,precision=.9,recall=.3))


def test_none_and_platt_calibration_are_finite_and_monotone():
    p = np.linspace(.01,.99,100)
    y = (p > .6).astype(int)
    model = opt.calibrator('platt',y,p)
    cp = opt.calibrated('platt',model,p)
    assert np.all(np.diff(cp)>=0)
    assert np.all((cp>=0)&(cp<=1))
    np.testing.assert_array_equal(opt.calibrated('none',None,p),p)


def test_development_refuses_to_overwrite_existing_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(opt,'OUT',tmp_path)
    with pytest.raises(FileExistsError,match='Refusing'):
        opt.develop()


def test_no_frozen_cv_after_test_has_started(tmp_path, monkeypatch):
    monkeypatch.setattr(opt,'ART',tmp_path)
    (tmp_path/'frozen.json').write_text('{}')
    (tmp_path/'TEST_EVALUATION_STARTED.json').write_text('{}')
    with pytest.raises(FileExistsError,match='precede test'):
        opt.validate_frozen()
