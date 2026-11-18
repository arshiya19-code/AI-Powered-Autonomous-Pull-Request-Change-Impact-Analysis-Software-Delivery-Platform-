"""Verify persisted experiment outputs without retraining or selecting models."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/risk_model'
    artifacts = ROOT / 'models/risk_model/base_paper_comparison'
    result = json.loads((reports / 'gating_metrics.json').read_text())
    frozen = json.loads((artifacts / 'frozen_models.json').read_text())
    for path, digest in result['protocol']['hashes_before'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    predictions = pd.read_parquet(reports / 'comparison_test_predictions.parquet')
    raw = pd.read_csv(ROOT / 'data/raw/apachejit/apachejit_total.csv').set_index('commit_id').loc[predictions.commit_id]
    assert np.array_equal(raw.buggy.astype(int), predictions.buggy)
    assert len(predictions) == 15736 and int(predictions.buggy.sum()) == 5143
    raw['files_changed'], raw['lines_added'], raw['lines_deleted'] = raw.nf, raw.la, raw.ld
    raw['total_lines_changed'] = raw.la + raw.ld
    base = result['protocol']['features'][:4]
    lr = joblib.load(artifacts / 'logistic_regression.joblib')
    v1, v2 = CatBoostClassifier(), CatBoostClassifier()
    v1.load_model(str(ROOT / 'models/risk_model/catboost_apachejit.cbm'))
    v2.load_model(str(artifacts / 'catboost_v2.cbm'))
    candidates = [('Logistic Regression', lr, base), ('CatBoost V1', v1, base),
                  ('CatBoost V2', v2, result['protocol']['features'])]
    for name, model, features in candidates:
        frame = raw[features].fillna(frozen['medians'])
        p = model.predict_proba(frame.to_numpy() if name == 'CatBoost V1' else frame)[:, 1]
        assert np.allclose(p, predictions[name], atol=1e-14, rtol=0), name
        # Independent persisted-table top-k check; ties retain saved row order.
        ranked = predictions.sort_values(name, ascending=False, kind='stable')
        for budget in [5, 10, 50]:
            gate = result['models'][name]['gating'][str(budget)]
            k = (len(predictions) * budget + 99) // 100
            captured = int(ranked.iloc[:k].buggy.sum())
            assert k == gate['gated_count'] and captured == gate['captured_count']
            assert captured / 5143 == gate['capture']
    verification = dict(original_hashes_verified=len(result['protocol']['hashes_before']),
                        saved_model_prediction_parity=True, independent_gated_counts=True,
                        unchanged_test_labels=True, test_rows=len(predictions))
    path = reports / 'comparison_verification.json'
    with path.open('x') as handle:
        json.dump(verification, handle, indent=2)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
