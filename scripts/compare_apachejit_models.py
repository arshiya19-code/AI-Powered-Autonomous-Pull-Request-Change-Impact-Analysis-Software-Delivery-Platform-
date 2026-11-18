"""Reproducible, additive ApacheJIT gating experiment; never deploys a model.

Run: python scripts/compare_apachejit_models.py --bootstrap 1000
Existing outputs are refused. All fitting and threshold selection precede test scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ['files_changed', 'lines_added', 'lines_deleted', 'total_lines_changed']
DIFFUSION = ['nd', 'ns', 'ent']
HISTORY = ['ndev', 'age', 'nuc']
AUTHOR = ['aexp', 'arexp', 'asexp']
EXPANDED = BASE + DIFFUSION + HISTORY + AUTHOR
BUDGETS = (5, 10, 50)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gating(y, p):
    """Input is in a pre-established, label-independent tie order; gate ceil(q*n)."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Invalid labels or probabilities')
    ranked = np.argsort(-p, kind='stable')
    positives = int(y.sum())
    result = {}
    for budget in BUDGETS:
        k = math.ceil(len(y) * budget / 100)
        captured = int(y[ranked[:k]].sum())
        capture = captured / positives if positives else None
        result[str(budget)] = dict(gated_count=k, captured_count=captured,
                                  positive_count=positives, actual_gating_fraction=k / len(y),
                                  capture=capture, lift=capture / (budget / 100) if positives else None)
    return result


def metrics(y, p, threshold):
    pred = p >= threshold
    return dict(threshold=float(threshold), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier_score=brier_score_loss(y, p), gating=gating(y, p))


def select_threshold(y, p):
    grid = np.linspace(.05, .95, 181)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def interval(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                valid_replicates=len(values)) if len(values) else None


def bootstrap(y, predictions, repeats):
    rng = np.random.default_rng(20260911)
    keys = ['roc_auc', 'pr_auc'] + [f'capture_{b}' for b in BUDGETS]
    samples = {name: {k: [] for k in keys} for name in predictions}
    ratios = {name: {str(b): [] for b in BUDGETS} for name in predictions if name != 'Logistic Regression'}
    differences = {name: {k: [] for k in keys} for name in ratios}
    for iteration in range(repeats):
        # Same resampled commits for all models. Sort indices to preserve fixed tie order.
        ix = np.sort(rng.integers(0, len(y), len(y)))
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            continue
        current = {}
        for name, p in predictions.items():
            pp = p[ix]
            gates = gating(yy, pp)
            current[name] = dict(roc_auc=roc_auc_score(yy, pp), pr_auc=average_precision_score(yy, pp),
                                 **{f'capture_{b}': gates[str(b)]['capture'] for b in BUDGETS})
            for k, v in current[name].items():
                samples[name][k].append(v)
        for name in ratios:
            for k in keys:
                differences[name][k].append(current[name][k] - current['Logistic Regression'][k])
            for b in BUDGETS:
                denominator = current['Logistic Regression'][f'capture_{b}']
                ratios[name][str(b)].append(current[name][f'capture_{b}'] / denominator if denominator else None)
        if (iteration + 1) % 100 == 0:
            print(f'Bootstrap {iteration + 1}/{repeats}', flush=True)
    return dict(method='paired IID commit percentile bootstrap; conditional on these three test projects; not project-population inference',
                seed=20260911, requested_replicates=repeats,
                models={n: {k: interval(v) for k, v in d.items()} for n, d in samples.items()},
                capture_ratios={n: {k: interval(v) for k, v in d.items()} for n, d in ratios.items()},
                differences_vs_logistic={n: {k: interval(v) for k, v in d.items()} for n, d in differences.items()})


def audit(raw):
    descriptions = dict(la='Lines added', ld='Lines deleted', nf='Files touched', nd='Directories touched',
                        ns='Subsystems touched', ent='Change entropy', ndev='Historical developers of changed files (dataset aggregate)',
                        age='Average time since prior file change', nuc='Historical unique file changes (dataset aggregate)',
                        aexp='Author prior experience', arexp='Author recent experience', asexp='Author subsystem experience')
    rows = []
    for col in raw.columns:
        selected = col in descriptions
        if col in ['buggy', 'fix', 'project']:
            classification = 'LEAKAGE'
            reason = {'buggy': 'Target only; SZZ outcome', 'fix': 'Excluded fixing indicator per experiment policy; may encode issue evidence',
                      'project': 'Group identity prohibited as predictor; split/audit only'}[col]
        elif col in ['commit_id', 'year', 'author_date']:
            classification = 'SAFE_AT_PREDICTION_TIME'
            reason = 'Excluded identity/time metadata; no outcome inference or proxy predictors'
        elif col in HISTORY + AUTHOR:
            classification = 'SAFE_AT_PREDICTION_TIME|REQUIRES_HISTORY|UNAVAILABLE_AT_RUNTIME'
            reason = 'Pre-change history by published dataset definition; runtime has no matching historical extractor. No reconstruction from test labels.'
        else:
            classification = 'SAFE_AT_PREDICTION_TIME|DERIVABLE_FROM_CURRENT_ANALYZER'
            reason = 'Current change counts already emitted' if col in ['nf', 'la', 'ld'] else 'Derivable from changed paths/per-file churn; exact ApacheJIT semantics not implemented or parity-verified in runtime'
            if col in DIFFUSION:
                classification += '|UNAVAILABLE_AT_RUNTIME'
        rows.append(dict(column=col, description=descriptions.get(col, col), classification=classification,
                         selected_v2=selected, runtime_compatible=col in ['nf', 'la', 'ld'], reason=reason,
                         missing_count=int(raw[col].isna().sum()), dtype=str(raw[col].dtype)))
    rows.append(dict(column='la+ld', description='Derived total_lines_changed', classification='SAFE_AT_PREDICTION_TIME|DERIVABLE_FROM_CURRENT_ANALYZER',
                     selected_v2=True, runtime_compatible=True, reason='Same exact sum as V1', missing_count=0, dtype='derived numeric'))
    return pd.DataFrame(rows)


def markdown_table(rows, columns):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    reports = ROOT / 'reports/risk_model'
    artifacts = ROOT / 'models/risk_model/base_paper_comparison'
    outputs = ['base_paper_comparison.md', 'gating_metrics.json', 'model_comparison.csv', 'feature_ablation.csv',
               'expanded_feature_audit.csv', 'comparison_test_predictions.parquet', 'comparison_project_metrics.csv']
    if artifacts.exists() or any((reports / name).exists() for name in outputs):
        raise FileExistsError('Refusing to overwrite comparison artifacts; archive them explicitly before rerunning')
    protected = [p for folder in [reports, ROOT / 'models/risk_model'] for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT / 'data/raw/apachejit/apachejit_total.csv', ROOT / 'data/processed/risk_model/apachejit_catboost_processed.parquet']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    original = json.loads((reports / 'metrics.json').read_text())
    metadata = json.loads((ROOT / 'models/risk_model/catboost_apachejit_metadata.json').read_text())
    preprocessing = json.loads((ROOT / 'models/risk_model/catboost_apachejit_preprocessing.json').read_text())
    d = pd.read_parquet(ROOT / 'data/processed/risk_model/apachejit_catboost_processed.parquet')
    raw = pd.read_csv(ROOT / 'data/raw/apachejit/apachejit_total.csv')
    assert d.commit_id.is_unique and raw.commit_id.is_unique and len(d) == len(raw)
    aligned = raw.set_index('commit_id').loc[d.commit_id].reset_index()
    assert np.array_equal(d.buggy, aligned.buggy.astype(int))
    assert np.array_equal(d.project, aligned.project)
    for dest, source in zip(BASE[:3], ['nf', 'la', 'ld']):
        assert np.array_equal(d[dest], aligned[source], equal_nan=True)
    assert np.array_equal(d.total_lines_changed, aligned.la + aligned.ld, equal_nan=True)
    for col in DIFFUSION + HISTORY + AUTHOR:
        d[col] = aligned[col].to_numpy()
    # Pseudorandom, reproducible label-independent ordering for equal-risk commits.
    d['_tie'] = d.commit_id.map(lambda x: hashlib.sha256(('apachejit-gating-42:' + x).encode()).hexdigest())
    d = d.sort_values('_tie').reset_index(drop=True)
    parts = {s: d[d.split == s].copy() for s in ['train', 'validation', 'test']}
    assert sum(map(len, parts.values())) == len(d)
    for s, part in parts.items():
        assert set(part.project) == set(original['split_strategy']['projects'][s])
        assert len(part) == original['split_strategy']['actual'][s]['rows']
        for other in parts:
            if s != other:
                assert set(part.project).isdisjoint(parts[other].project)
    train, val, test = (parts[s] for s in ['train', 'validation', 'test'])
    assert np.isfinite(train[EXPANDED].dropna().to_numpy()).all()
    artifacts.mkdir(parents=True)
    protocol = dict(seed=42, features=EXPANDED, threshold_rule='maximum validation F1 on 0.05..0.95 step 0.005; smallest threshold on ties',
                    logistic='median imputation + StandardScaler fitted on train; L2 C=1, lbfgs, no class weighting, max_iter=2000; no tuning',
                    catboost='same preset as V1: 500 iterations, depth 6, lr .05, l2 3, train class weight; validation PRAUC early stopping 50',
                    ablations='fixed in advance: fresh four-feature control; add diffusion; add history; add author; all expanded. No test selection.',
                    test_policy='test predictions only after all fits and thresholds frozen; prior V1 test exposure acknowledged',
                    hashes_before=before, versions=dict(python=platform.python_version(), numpy=np.__version__,
                                                        pandas=pd.__version__, sklearn=sklearn.__version__, catboost=catboost.__version__))
    (artifacts / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    models, thresholds, feature_sets = {}, {}, {}
    lr = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                       LogisticRegression(C=1, max_iter=2000, random_state=42))
    lr.fit(train[BASE], train.buggy)
    assert max(lr[-1].n_iter_) < 2000
    models['Logistic Regression'] = lr
    feature_sets['Logistic Regression'] = BASE
    thresholds['Logistic Regression'] = select_threshold(val.buggy, lr.predict_proba(val[BASE])[:, 1])
    joblib.dump(lr, artifacts / 'logistic_regression.joblib')
    v1 = CatBoostClassifier()
    v1.load_model(str(ROOT / 'models/risk_model/catboost_apachejit.cbm'))
    assert metadata['feature_order'] == BASE
    assert v1.feature_names_ in [BASE, ['0', '1', '2', '3']]
    models['CatBoost V1'] = v1
    feature_sets['CatBoost V1'] = BASE
    thresholds['CatBoost V1'] = metadata['selected_threshold']
    medians = train[EXPANDED].median()
    for f in BASE:
        assert medians[f] == preprocessing['training_medians'][f]
    presets = {'CatBoost control': BASE, 'CatBoost +diffusion': BASE + DIFFUSION,
               'CatBoost +history': BASE + DIFFUSION + HISTORY,
               'CatBoost +author': BASE + DIFFUSION + AUTHOR, 'CatBoost V2': EXPANDED}
    fit_info = {}
    for name, features in presets.items():
        print(f'Training {name}: {len(features)} features', flush=True)
        model = CatBoostClassifier(iterations=500, depth=6, learning_rate=.05, l2_leaf_reg=3,
                                   loss_function='Logloss', eval_metric='PRAUC', random_seed=42,
                                   scale_pos_weight=float((train.buggy == 0).sum() / train.buggy.sum()),
                                   thread_count=4, allow_writing_files=False, verbose=False)
        model.fit(train[features].fillna(medians), train.buggy,
                  eval_set=(val[features].fillna(medians), val.buggy), early_stopping_rounds=50)
        pval = model.predict_proba(val[features].fillna(medians))[:, 1]
        thresholds[name] = select_threshold(val.buggy, pval)
        models[name], feature_sets[name] = model, features
        fit_info[name] = dict(features=features, tree_count=model.tree_count_,
                              validation=metrics(val.buggy.to_numpy(), pval, thresholds[name]))
        model.save_model(str(artifacts / (name.lower().replace(' ', '_').replace('+', '') + '.cbm')))
    (artifacts / 'frozen_models.json').write_text(json.dumps(dict(thresholds=thresholds, fits=fit_info,
                                                               medians=medians.to_dict(), runtime_model='existing CatBoost V1',
                                                               v2_status='RESEARCH / BENCHMARK MODEL'), indent=2))
    print('All models and thresholds frozen. Scoring untouched test partition.', flush=True)
    y = test.buggy.to_numpy()
    predictions = {name: model.predict_proba(test[feature_sets[name]].fillna(medians).to_numpy()
                                            if name == 'CatBoost V1' else test[feature_sets[name]].fillna(medians))[:, 1]
                   for name, model in models.items()}
    results = {name: metrics(y, p, thresholds[name]) for name, p in predictions.items()}
    for key in ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier_score']:
        assert np.isclose(results['CatBoost V1'][key], original['catboost_test_metrics'][key], atol=1e-12), key
    principal = ['Logistic Regression', 'CatBoost V1', 'CatBoost V2']
    rows = []
    for name, result in results.items():
        row = dict(model=name, features=len(feature_sets[name]), feature_names=';'.join(feature_sets[name]),
                   **{k: v for k, v in result.items() if k != 'gating'})
        for b in BUDGETS:
            g = result['gating'][str(b)]
            baseline = results['Logistic Regression']['gating'][str(b)]['capture']
            row.update({f'capture_{b}': g['capture'], f'lift_{b}': g['lift'],
                        f'relative_{b}': g['capture'] / baseline if baseline else None})
        rows.append(row)
    project_rows = []
    for project in sorted(test.project.unique()):
        ix = (test.project == project).to_numpy()
        for name in principal:
            m = metrics(y[ix], predictions[name][ix], thresholds[name])
            project_rows.append(dict(project=project, model=name, rows=int(ix.sum()), positives=int(y[ix].sum()),
                                     **{k: v for k, v in m.items() if k != 'gating'},
                                     **{f'capture_{b}': m['gating'][str(b)]['capture'] for b in BUDGETS}))
    ci = bootstrap(y, {n: predictions[n] for n in principal}, args.bootstrap)
    assert all(sha(ROOT / p) == digest for p, digest in before.items()), 'Existing artifact changed'
    payload = dict(protocol=protocol, original_artifacts_unchanged=True, test_rows=len(test), test_positives=int(y.sum()),
                   gating_policy='descending probability; ceil(N*q); stable SHA256(commit_id) tie order; lift uses nominal q',
                   pr_auc_definition='average precision (non-interpolated), not trapezoidal PR area',
                   models=results, confidence_intervals=ci, project_metrics=project_rows)
    (reports / 'gating_metrics.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    pd.DataFrame([r for r in rows if r['model'] in principal]).to_csv(reports / 'model_comparison.csv', index=False)
    pd.DataFrame([r for r in rows if r['model'] in presets]).to_csv(reports / 'feature_ablation.csv', index=False)
    audit(raw).to_csv(reports / 'expanded_feature_audit.csv', index=False)
    pd.DataFrame(project_rows).to_csv(reports / 'comparison_project_metrics.csv', index=False)
    saved = test[['commit_id', 'project', 'buggy']].copy()
    for name, p in predictions.items():
        saved[name] = p
    saved.to_parquet(reports / 'comparison_test_predictions.parquet', index=False)
    primary_rows = [r for r in rows if r['model'] in principal]
    report = ['# ApacheJIT base-paper comparison',
              'We adopt the same risk-ranking/gating evaluation principle.',
              'The [Meta paper](https://arxiv.org/html/2410.06351v1) predicts severe production incidents (SEVs); ApacheJIT predicts SZZ-labeled bug-inducing commits. Absolute numbers are not directly comparable. The paper reports logistic capture of 18.7%, 27.9%, 84.6% at 5%, 10%, 50% gating; risk-aligned iDiffLlama relative capture is 1.40x, 1.52x, 1.05x. This experiment compares methodology and improvement over our own logistic baseline, not numerical superiority to Meta.',
              '## Protocol and preservation',
              'All original artifacts, raw data, processed rows, labels, splits and V1 threshold (0.565) are SHA256-verified unchanged. V1 metrics reproduce to 1e-12. Exact saved project groups are reused: train 75,053; validation 15,885; test 15,736 (5,143 positives). No project identity is predictive. This is a project-isolated, not chronological, experiment. The existing split was previously selected by a group-split candidate search; it is not reselected here. Test data had already been evaluated for V1, so it is a reused benchmark holdout, not a newly blinded external test.',
              'Logistic Regression is newly fitted using exactly the four V1 features, training-only median imputation and standard scaling, L2 C=1, no class weights or hyperparameter search. New thresholds maximize validation F1 on a fixed grid. V2 uses the V1 training preset and validation-only early stopping. All thresholds and models were saved before test scoring. No model is selected or deployed based on test results.',
              'Gate ceil(N*q) commits in descending probability order. Equal probabilities use a fixed SHA256 commit ordering independent of labels and projects. Capture = gated positives / all test positives; lift = capture / nominal q. JSON includes exact counts and realized budgets. PR-AUC means average precision. Numeric CSV/JSON values retain full floating-point precision.',
              '## Main comparison', markdown_table(primary_rows, ['model', 'features', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier_score', 'threshold']),
              markdown_table(primary_rows, ['model', 'capture_5', 'capture_10', 'capture_50', 'lift_5', 'lift_10', 'lift_50', 'relative_5', 'relative_10', 'relative_50']),
              '## Feature and leakage audit',
              'V2 selects 13 inputs: ' + ', '.join(EXPANDED) + '. The first four map to nf, la, ld and la+ld. Runtime remains V1. V2 is **RESEARCH / BENCHMARK MODEL**. nd/ns/ent can be derived from current changed paths and per-file churn, but are not currently emitted with verified ApacheJIT semantics. ndev/age/nuc/aexp/arexp/asexp require historical repository state and have no corresponding runtime extractor.',
              'The [ApacheJIT dataset paper, Sections 3 and 4.5](https://arxiv.org/html/2203.00101) describes these as current-change and prior-history metrics. Historical values are used as supplied, not reconstructed using held-out labels. Their prediction-time validity relies on the published extraction definitions; this run does not independently remine repositories to prove every timestamp boundary. Fractional ndev/nuc/asexp values are preserved as dataset aggregates. No invented lt/exp/rexp/sexp columns are added.',
              'Excluded: buggy as predictor, fix, project identity, commit identifier, year and author_date. Future defect evidence, future issue links, future timestamps and outcomes are never inputs. Timestamps and identifiers only support alignment/audit. expanded_feature_audit.csv covers every raw column and the derived sum. Dataset curation and SZZ label uncertainty remain limitations; exclusion of predictors does not undo benchmark sampling bias.',
              '## Confidence intervals',
              f'{args.bootstrap} paired IID commit bootstrap resamples, seed 20260911, percentile 95% intervals. Same resampled commits for every model, reranking within each sample. These quantify uncertainty conditional on these projects and fixed trained models; commits may be correlated, so intervals can be optimistic. They do not establish generalization to a population of projects. There are only three held-out projects. Intervals are pointwise, without multiple-comparison correction.',
              markdown_table([dict(model=n, metric=k, low=v['low'], high=v['high']) for n, items in ci['models'].items() for k, v in items.items()], ['model', 'metric', 'low', 'high']),
              'Paired differences versus Logistic Regression (positive intervals favor CatBoost):',
              markdown_table([dict(model=n, metric=k, low=v['low'], high=v['high']) for n, items in ci['differences_vs_logistic'].items() for k, v in items.items()], ['model', 'metric', 'low', 'high']),
              'Paired capture-ratio intervals:',
              markdown_table([dict(model=n, budget=k, low=v['low'], high=v['high']) for n, items in ci['capture_ratios'].items() for k, v in items.items()], ['model', 'budget', 'low', 'high']),
              '## Held-out project consistency', markdown_table(project_rows, ['project', 'model', 'rows', 'roc_auc', 'pr_auc', 'capture_5', 'capture_10', 'capture_50']),
              '## Prespecified feature ablations',
              'These are additive controls with identical training presets; early stopping and threshold selection remain validation-only. The fresh four-feature control differs from preserved V1 in training row order; use it to isolate feature additions under the same implementation. These are descriptive experiments, not test-driven selection.',
              markdown_table([r for r in rows if r['model'] in presets], ['model', 'features', 'f1', 'roc_auc', 'pr_auc', 'capture_5', 'capture_10', 'capture_50']),
              '## Conclusion']
    for name in principal[1:]:
        overall = [b for b in BUDGETS if results[name]['gating'][str(b)]['capture'] > results['Logistic Regression']['gating'][str(b)]['capture']]
        significant = [b for b in BUDGETS if ci['differences_vs_logistic'][name][f'capture_{b}']['low'] > 0]
        wins = 0
        for row in project_rows:
            if row['model'] == name:
                ref = next(r for r in project_rows if r['project'] == row['project'] and r['model'] == 'Logistic Regression')
                wins += sum(row[f'capture_{b}'] > ref[f'capture_{b}'] for b in BUDGETS)
        report.append(f'{name}: higher pooled capture at budgets {overall}; positive pointwise paired 95% difference intervals at {significant}; wins {wins}/9 project-budget comparisons. ' +
                      ('Consistent across all evaluated project-budget cells, subject to the three-project limitation.' if wins == 9 else 'Not a consistent improvement across projects and budgets; the stated consistency criterion is not met.'))
    report += ['Runtime preference remains the existing four-feature CatBoost V1; V2 is retained only as a research artifact. No ROC-AUC target was imposed.',
               '## Reproduction and artifacts',
               'Run `python scripts/compare_apachejit_models.py --bootstrap 1000` in a compatible environment. The script refuses existing comparison outputs. Model files, fitted logistic pipeline, frozen thresholds, versions and input hashes are under models/risk_model/base_paper_comparison/. Full metrics and bootstrap intervals are in gating_metrics.json; predictions and project results are saved alongside the requested CSV files. Existing reports were not rewritten.',
               'BASE-PAPER COMPARISON EXPERIMENT COMPLETE']
    (reports / 'base_paper_comparison.md').write_text('\n\n'.join(report), encoding='utf-8')
    print(markdown_table(primary_rows, ['model', 'f1', 'roc_auc', 'pr_auc', 'capture_5', 'capture_10', 'capture_50']), flush=True)


if __name__ == '__main__':
    main()
