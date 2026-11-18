"""One bounded development search, explicit freeze, then one final test evaluation.

Run --develop first; --finalize consumes the immutable freeze without any fitting.
Neither command overwrites outputs. Historical benchmark values are never deployed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from compare_apachejit_models import (ROOT, BASE, DIFFUSION, HISTORY, AUTHOR,
    EXPANDED, metrics, audit, sha, markdown_table, interval)

OUT = ROOT / 'reports/risk_model/final_multimetric'
ART = ROOT / 'models/risk_model/final_multimetric'
KEYS = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier_score',
        'capture_5', 'capture_10', 'capture_50', 'lift_5', 'lift_10', 'lift_50']
DERIVED = ['add_delete_ratio', 'churn_per_file', 'churn_per_directory',
           'entropy_per_file', 'recent_experience_ratio', 'subsystem_experience_ratio',
           'developer_density']


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def flat(y, p, t):
    m = metrics(np.asarray(y), np.asarray(p), t)
    for b, v in m.pop('gating').items():
        m['capture_' + b], m['lift_' + b] = v['capture'], v['lift']
    return m


def objective(m):
    return (.25*m['roc_auc'] + .25*m['pr_auc'] + .20*m['f1']
            + .15*m['precision'] + .15*m['recall']
            - .05*abs(m['precision']-m['recall']))


def thresholds(y, p):
    # Sorted cumulative counts avoid fitting or rescoring 199 full prediction arrays.
    grid = np.linspace(.005, .995, 199)
    order = np.argsort(p, kind='stable')
    sy = np.asarray(y)[order]
    cum = np.r_[0, np.cumsum(sy)]
    start = np.searchsorted(np.asarray(p)[order], grid, side='left')
    tp = cum[-1] - cum[start]
    precision = tp / np.maximum(len(y)-start, 1)
    recall = tp / max(cum[-1], 1)
    f1 = 2*precision*recall / np.maximum(precision+recall, 1e-15)
    balanced = .4*f1 + .3*precision + .3*recall - .1*np.abs(precision-recall)
    def pick(score, mask=None):
        if mask is not None and not np.any(mask):
            return None
        i = int(np.argmax(score if mask is None else np.where(mask, score, -np.inf)))
        return dict(threshold=float(grid[i]), precision=float(precision[i]),
                    recall=float(recall[i]), f1=float(f1[i]), balanced_score=float(balanced[i]))
    return dict(max_f1=pick(f1), balanced=pick(balanced),
                high_recall=pick(precision, recall >= .80),
                high_precision=pick(recall, precision >= .80))


def derive(d):
    x = d[EXPANDED].copy()
    x['add_delete_ratio'] = x.lines_added / (x.lines_deleted + 1)
    x['churn_per_file'] = x.total_lines_changed / x.files_changed.clip(lower=1)
    x['churn_per_directory'] = x.total_lines_changed / x.nd.clip(lower=1)
    x['entropy_per_file'] = x.ent / x.files_changed.clip(lower=1)
    x['recent_experience_ratio'] = x.arexp / (x.aexp + 1)
    x['subsystem_experience_ratio'] = x.asexp / (x.aexp + 1)
    x['developer_density'] = x.ndev / x.files_changed.clip(lower=1)
    return x


def matrix(d, spec):
    x = derive(d)[spec['source_features']].copy()
    for f in spec['logs']:
        if spec['transform'] == 'log':
            x[f] = np.log1p(x[f])
        elif spec['transform'] == 'both':
            x['log1p_' + f] = np.log1p(x[f])
    if spec.get('keep'):
        x = x[spec['keep']]
    return x


def load_data(test=False):
    # Partition filtering happens before aligning or inspecting feature distributions.
    d = pd.read_parquet(ROOT / 'data/processed/risk_model/apachejit_catboost_processed.parquet')
    assert d.commit_id.is_unique
    groups = {s: set(d.loc[d.split == s, 'project']) for s in ['train', 'validation', 'test']}
    assert all(groups[a].isdisjoint(groups[b]) for a, b in [('train','validation'),('train','test'),('validation','test')])
    expected = json.loads((ROOT/'reports/risk_model/metrics.json').read_text())['split_strategy']['projects']
    assert all(groups[s] == set(expected[s]) for s in groups)
    d = d.loc[d.split.eq('test') if test else d.split.ne('test')].copy()
    raw = pd.read_csv(ROOT / 'data/raw/apachejit/apachejit_total.csv')
    assert raw.commit_id.is_unique
    raw = raw.set_index('commit_id').loc[d.commit_id].reset_index()
    assert np.array_equal(d.buggy, raw.buggy.astype(int))
    assert np.array_equal(d.project, raw.project)
    for dest, src in zip(BASE[:3], ['nf','la','ld']):
        assert np.array_equal(d[dest], raw[src], equal_nan=True)
    assert np.array_equal(d.total_lines_changed, raw.la + raw.ld, equal_nan=True)
    for c in DIFFUSION+HISTORY+AUTHOR:
        d[c] = raw[c].to_numpy()
    assert np.isin(d.buggy, [0,1]).all()
    d['_tie'] = d.commit_id.map(lambda s: hashlib.sha256(('apachejit-gating-42:'+s).encode()).hexdigest())
    return d.sort_values('_tie').reset_index(drop=True), raw


def fit(train, val, spec):
    x, v = matrix(train, spec), matrix(val, spec)
    med = x.median().fillna(0)
    assert np.isfinite(x.fillna(med)).all().all()
    m = CatBoostClassifier(**spec['params'], loss_function='Logloss', eval_metric='PRAUC',
                           random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
    m.fit(x.fillna(med), train.buggy, eval_set=(v.fillna(med),val.buggy), early_stopping_rounds=60)
    return m, med, m.predict_proba(v.fillna(med))[:,1]


def calibrator(kind, y, p):
    if kind == 'none':
        return None
    if kind == 'platt':
        return LogisticRegression(C=1e6, max_iter=1000).fit(logit(p), y)
    return IsotonicRegression(out_of_bounds='clip').fit(p, y)


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def calibrated(kind, model, p):
    return p if kind == 'none' else (model.predict_proba(logit(p))[:,1] if kind == 'platt' else model.predict(p))


def protected_hashes():
    paths = [ROOT/'data/raw/apachejit/apachejit_total.csv', ROOT/'data/processed/risk_model/apachejit_catboost_processed.parquet']
    paths += [p for folder in ['models/risk_model','reports/risk_model'] for p in (ROOT/folder).rglob('*')
              if p.is_file() and 'final_multimetric' not in p.parts]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def develop():
    if OUT.exists() or ART.exists():
        raise FileExistsError('Refusing to repeat or overwrite development phase')
    OUT.mkdir(parents=True); ART.mkdir(parents=True)
    protocol = dict(seed=42, catboost_configurations=40,
        objective='.25 ROC-AUC + .25 average precision + .20 F1 + .15 precision + .15 recall - .05 abs(precision-recall)',
        threshold='199 thresholds .005 to .995; maximize .4 F1 + .3 precision + .3 recall - .1 absolute imbalance; smallest on ties; high targets .80',
        selection='Top 3 validation objectives receive 3 group-aware outer CV folds on training projects; select .5 validation objective + .5 CV mean objective - .1 CV objective SD, plus .01 mean validation capture@5/@10, minus .02 validation Brier',
        cv='Each outer training set uses a deterministic group-disjoint 25% inner holdout for early stopping and threshold selection. Outer labels never tune thresholds. CV is a stability diagnostic, not unbiased nested search evaluation: shortlist already used all training labels.',
        calibration='Leave-one-validation-project-out calibration; compare none, Platt, isotonic on pooled cross-fitted predictions; final calibrator fitted on all validation projects only after selection.',
        variants='Logistic weights 0,.1,.2,.3. Accept blend only if objective rises and >=2 of five primary metrics rise, none falls >.01, capture@5/@10 do not fall >.01, Brier does not rise >.005. Calibration additionally must lower Brier and preserve ranking within .005.',
        final_test='One locked evaluation after freeze; no retuning. This test was exposed in previous research phases; cannot call it a new untouched confirmation set.',
        temporal='Original project split preserved; not a prospective temporal split. Historical predictors rely on published pre-change definitions; no new future-derived predictors or label edits.',
        source='https://arxiv.org/html/2203.00101',
        versions=dict(python=platform.python_version(), catboost=catboost.__version__, sklearn=sklearn.__version__, pandas=pd.__version__),
        hashes_before=protected_hashes())
    save(ART/'protocol.json', protocol)
    d, raw = load_data()
    train, val = d[d.split.eq('train')], d[d.split.eq('validation')]
    audit(raw).to_csv(OUT/'feature_audit.csv', index=False)
    x = derive(train)
    assert not (x.dropna() < 0).any().any()
    stats = pd.DataFrame({'skew':x.skew(), 'median':x.median(), 'p95':x.quantile(.95), 'p99':x.quantile(.99), 'max':x.max(), 'missing':x.isna().sum()})
    stats.to_csv(OUT/'training_distributions.csv')
    skewed = stats.index[stats['skew'] > 2].tolist()
    groups = [BASE, BASE+DIFFUSION, BASE+DIFFUSION+HISTORY, BASE+DIFFUSION+AUTHOR, EXPANDED]
    preset = dict(iterations=700, depth=6, learning_rate=.05, l2_leaf_reg=3,
                  random_strength=1, bootstrap_type='Bayesian', bagging_temperature=1,
                  border_count=128, grow_policy='SymmetricTree')
    specs=[]
    for i, fs in enumerate(groups):
        specs.append(dict(name=['base4','diffusion','history','author','all_safe'][i], source_features=fs,
                          transform='raw', logs=[], params=preset.copy()))
    for transform in ['raw','log','both']:
        fs = EXPANDED+DERIVED
        specs.append(dict(name='derived_'+transform, source_features=fs, transform=transform,
                          logs=[f for f in fs if f in skewed], params=preset.copy()))
    rng = np.random.default_rng(42)
    for i in range(30):
        fs = [EXPANDED, EXPANDED+DERIVED, BASE+DIFFUSION+HISTORY][i%3]
        tr = ['raw','log','both'][(i//3)%3]
        par = dict(iterations=int(rng.choice([600,1000,1400])), depth=int(rng.choice([4,5,6,7,8])),
                   learning_rate=float(rng.choice([.025,.04,.06,.08])), l2_leaf_reg=float(rng.choice([3,7,15,30])),
                   random_strength=float(rng.choice([.2,1,2])), border_count=int(rng.choice([64,128,254])),
                   grow_policy=['SymmetricTree','Depthwise','Lossguide'][i%3])
        if par['grow_policy'] != 'SymmetricTree':
            par['min_data_in_leaf'] = int(rng.choice([20,50,100]))
        if par['grow_policy'] == 'Lossguide':
            par['max_leaves'] = 31
        par['bootstrap_type'] = ['Bayesian','Bernoulli'][i%2]
        if par['bootstrap_type'] == 'Bayesian':
            par['bagging_temperature'] = float(rng.choice([0,.5,1,3]))
        else:
            par['subsample'] = float(rng.choice([.7,.9,1]))
        if i%4 == 1:
            par['auto_class_weights'] = 'Balanced'
        elif i%4 >= 2:
            par['class_weights'] = [1, [1.5,2][i%4-2]]
        specs.append(dict(name=f'search_{i:02}', source_features=fs, transform=tr,
                          logs=[f for f in fs if f in skewed], params=par))
    models, medians, preds, rows = {}, {}, {}, []
    def run(spec):
        i = len(rows)
        print(f'Configuration {i+1}/40: {spec["name"]}', flush=True)
        m, med, p = fit(train, val, spec)
        ts = thresholds(val.buggy.to_numpy(),p)
        met = flat(val.buggy,p,ts['balanced']['threshold'])
        row = dict(id=i,name=spec['name'],trees=m.tree_count_,objective=objective(met),**met)
        rows.append(row); models[i]=m; medians[i]=med; preds[i]=p
        m.save_model(str(ART/f'candidate_{i:02}.cbm'))
        save(ART/f'candidate_{i:02}.json',dict(spec=spec,medians=med.to_dict(),thresholds=ts,validation=row))
        pd.DataFrame(rows).to_csv(OUT/'validation_search.csv',index=False)
        print(f'Validation objective={row["objective"]:.4f}, F1={met["f1"]:.4f}, AUC={met["roc_auc"]:.4f}, AP={met["pr_auc"]:.4f}',flush=True)
    for spec in specs:
        run(spec)
    best = max(rows,key=lambda r:r['objective'])['id']
    best_spec = specs[best]
    imp = pd.Series(models[best].feature_importances_,index=matrix(train,best_spec).columns).sort_values(ascending=False)
    imp.to_csv(OUT/'catboost_importance.csv',header=['importance'])
    # Pruning is prespecified to two sizes and uses training-learned model importance.
    for count in [8,12]:
        spec = dict(best_spec, name=f'pruned_top_{count}', keep=imp.index[:count].tolist())
        specs.append(spec); run(spec)
    shortlist = sorted(rows,key=lambda r:r['objective'],reverse=True)[:3]
    cv_rows=[]; summaries=[]
    for candidate in shortlist:
        i=candidate['id']; spec=specs[i]
        for fold,(ti,vi) in enumerate(GroupKFold(3).split(train,train.buggy,train.project)):
            outer_train, outer = train.iloc[ti],train.iloc[vi]
            a,b = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=420+fold).split(outer_train,groups=outer_train.project))
            inner_train,inner_val=outer_train.iloc[a],outer_train.iloc[b]
            assert set(inner_train.project).isdisjoint(inner_val.project)
            assert set(outer_train.project).isdisjoint(outer.project)
            print(f'CV candidate {i}, fold {fold+1}/3',flush=True)
            m,med,p=fit(inner_train,inner_val,spec)
            t=thresholds(inner_val.buggy.to_numpy(),p)['balanced']['threshold']
            met=flat(outer.buggy,m.predict_proba(matrix(outer,spec).fillna(med))[:,1],t)
            cv_rows.append(dict(id=i,fold=fold,projects=';'.join(sorted(set(outer.project))),objective=objective(met),**met))
        cr=pd.DataFrame([r for r in cv_rows if r['id']==i])
        mean,sd=cr.objective.mean(),cr.objective.std(ddof=1)
        summaries.append(dict(id=i,cv_mean=float(mean),cv_sd=float(sd),selection_score=float(.5*candidate['objective']+.5*mean-.1*sd+.01*np.mean([candidate['capture_5'],candidate['capture_10']])-.02*candidate['brier_score'])))
    pd.DataFrame(cv_rows).to_csv(OUT/'cv_folds.csv',index=False)
    save(OUT/'cv_selection.json',summaries)
    chosen=max(summaries,key=lambda r:r['selection_score'])['id']
    spec=specs[chosen]; p=preds[chosen]
    lr=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42))
    lr.fit(train[BASE],train.buggy)
    lp=lr.predict_proba(val[BASE])[:,1]
    joblib.dump(lr,ART/'blend_logistic.joblib')
    y=val.buggy.to_numpy()
    variants=[]; vp={}
    for weight in [0,.1,.2,.3]:
        blend=(1-weight)*p+weight*lp
        for kind in ['none','platt','isotonic']:
            cp=blend.copy()
            if kind!='none':
                for project in sorted(set(val.project)):
                    hold=val.project.eq(project).to_numpy()
                    cal=calibrator(kind,y[~hold],blend[~hold])
                    cp[hold]=calibrated(kind,cal,blend[hold])
            ts=thresholds(y,cp); met=flat(y,cp,ts['balanced']['threshold'])
            key=f'{weight}_{kind}';vp[key]=(cp,ts)
            variants.append(dict(key=key,weight=weight,calibration=kind,objective=objective(met),**met))
    primary=['roc_auc','pr_auc','f1','precision','recall']
    native=variants[0]
    def blend_ok(r):
        return (r['weight']==0 or (r['objective']>native['objective'] and
                sum(r[k]>native[k]+1e-8 for k in primary)>=2 and
                all(r[k]>=native[k]-.01 for k in primary) and
                all(r[k]>=native[k]-.01 for k in ['capture_5','capture_10']) and
                r['brier_score']<=native['brier_score']+.005))
    eligible=[]
    for r in variants:
        base=next(v for v in variants if v['weight']==r['weight'] and v['calibration']=='none')
        cal_ok=r['calibration']=='none' or (r['brier_score']<base['brier_score'] and r['objective']>=base['objective'] and all(r[k]>=base[k]-.005 for k in ['roc_auc','pr_auc']))
        r['eligible']=bool(blend_ok(r) and cal_ok)
        if r['eligible']: eligible.append(r)
    winner=max(eligible,key=lambda r:r['objective']-.02*r['brier_score'])
    pd.DataFrame(variants).to_csv(OUT/'blend_calibration.csv',index=False)
    blend=(1-winner['weight'])*p+winner['weight']*lp
    cal=calibrator(winner['calibration'],y,blend)
    joblib.dump(cal,ART/'calibrator.joblib')
    models[chosen].save_model(str(ART/'final_catboost.cbm'))
    # Explanations inspect development data only; no further search follows.
    vx=matrix(val,spec).fillna(medians[chosen])
    sample=vx.sample(n=min(1000,len(vx)),random_state=42)
    shap=models[chosen].get_feature_importance(Pool(sample),type='ShapValues')[:,:-1]
    pd.DataFrame({'feature':vx.columns,'mean_abs_shap':np.abs(shap).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False).to_csv(OUT/'shap_importance.csv',index=False)
    perm=[];gen=np.random.default_rng(42)
    base_score=objective(flat(y,p,rows[chosen]['threshold']))
    for col in vx.columns:
        drops=[]
        for _ in range(2):
            z=vx.copy();z[col]=gen.permutation(z[col].to_numpy())
            drops.append(base_score-objective(flat(y,models[chosen].predict_proba(z)[:,1],rows[chosen]['threshold'])))
        perm.append(dict(feature=col,mean_objective_drop=float(np.mean(drops)),sd=float(np.std(drops))))
    pd.DataFrame(perm).sort_values('mean_objective_drop',ascending=False).to_csv(OUT/'permutation_importance.csv',index=False)
    selected_cv=pd.DataFrame([r for r in cv_rows if r['id']==chosen])
    cv_summary={k:dict(mean=float(selected_cv[k].mean()),variance=float(selected_cv[k].var(ddof=1)),sd=float(selected_cv[k].std(ddof=1))) for k in KEYS+['objective']}
    freeze=dict(candidate=chosen,spec=spec,feature_order=vx.columns.tolist(),medians=medians[chosen].to_dict(),
                trees=models[chosen].tree_count_,variant=winner,thresholds=vp[winner['key']][1],
                deployment_threshold=winner['threshold'],cv=cv_summary,
                cv_scope='Selected base CatBoost; optional blend/calibration evaluated by validation-project cross-fitting, not outer CV.',
                runtime='Unchanged existing CatBoost V1; this artifact is research-only.',
                hashes={p.name:sha(p) for p in [ART/'final_catboost.cbm',ART/'blend_logistic.joblib',ART/'calibrator.joblib',ART/f'candidate_{chosen:02}.json']})
    assert protected_hashes()==protocol['hashes_before']
    save(ART/'frozen.json',freeze)
    print('DEVELOPMENT COMPLETE; model and deployment threshold frozen. No test evaluation performed.',flush=True)


def validate_frozen():
    """Diagnostic CV of the frozen complete pipeline; never changes selection."""
    freeze=json.loads((ART/'frozen.json').read_text())
    if (ART/'TEST_EVALUATION_STARTED.json').exists() or (OUT/'frozen_pipeline_cv.json').exists():
        raise FileExistsError('Frozen pipeline diagnostics must precede test and run once')
    fitted=CatBoostClassifier();fitted.load_model(str(ART/'final_catboost.cbm'))
    save(ART/'effective_hyperparameters.json',fitted.get_all_params())
    variant=freeze['variant']
    if variant['weight']==0 and variant['calibration']=='none':
        save(OUT/'frozen_pipeline_cv.json',dict(summary=freeze['cv'],scope='Final pipeline is identical to the selected base CatBoost; reuse its existing outer CV.'))
        return
    d,_=load_data(); train=d[d.split.eq('train')]; rows=[]
    for fold,(ti,vi) in enumerate(GroupKFold(3).split(train,train.buggy,train.project)):
        development,outer=train.iloc[ti],train.iloc[vi]
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=420+fold).split(development,groups=development.project))
        inner,hold=development.iloc[a],development.iloc[b]
        print(f'Frozen complete pipeline CV fold {fold+1}/3',flush=True)
        model,med,p=fit(inner,hold,freeze['spec'])
        op=model.predict_proba(matrix(outer,freeze['spec']).fillna(med))[:,1]
        w=variant['weight']
        if w:
            lr=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42))
            lr.fit(inner[BASE],inner.buggy)
            p=(1-w)*p+w*lr.predict_proba(hold[BASE])[:,1]
            op=(1-w)*op+w*lr.predict_proba(outer[BASE])[:,1]
        kind=variant['calibration']; cp=p.copy(); y=hold.buggy.to_numpy()
        if kind!='none':
            assert hold.project.nunique()>=2
            for g in sorted(set(hold.project)):
                mask=hold.project.eq(g).to_numpy()
                cal=calibrator(kind,y[~mask],p[~mask])
                cp[mask]=calibrated(kind,cal,p[mask])
            cal=calibrator(kind,y,p)
            op=calibrated(kind,cal,op)
        t=thresholds(y,cp)['balanced']['threshold']
        m=flat(outer.buggy,op,t)
        rows.append(dict(fold=fold,projects=';'.join(sorted(set(outer.project))),objective=objective(m),**m))
    df=pd.DataFrame(rows);df.to_csv(OUT/'frozen_pipeline_cv_folds.csv',index=False)
    summary={k:dict(mean=float(df[k].mean()),variance=float(df[k].var(ddof=1)),sd=float(df[k].std(ddof=1))) for k in KEYS+['objective']}
    save(OUT/'frozen_pipeline_cv.json',dict(summary=summary,scope='Complete selected blend/calibration recipe; inner holdout selects thresholds with project-cross-fitted calibration. Diagnostic only: frozen model and selection unchanged. Post-selection CV, not unbiased nested search evaluation.'))


def finalize():
    freeze=json.loads((ART/'frozen.json').read_text())
    protocol=json.loads((ART/'protocol.json').read_text())
    json.loads((OUT/'frozen_pipeline_cv.json').read_text())
    for name,h in freeze['hashes'].items(): assert sha(ART/name)==h
    assert protected_hashes()==protocol['hashes_before']
    # Exclusive marker prevents even an accidental second evaluation.
    with (ART/'TEST_EVALUATION_STARTED.json').open('x') as f:
        json.dump(dict(freeze_sha256=sha(ART/'frozen.json'),script_sha256=sha(Path(__file__))),f)
    d,_=load_data(test=True)
    model=CatBoostClassifier();model.load_model(str(ART/'final_catboost.cbm'))
    p=model.predict_proba(matrix(d,freeze['spec']).fillna(pd.Series(freeze['medians'])))[:,1]
    variant=freeze['variant']; w=variant['weight']
    if w:
        p=(1-w)*p+w*joblib.load(ART/'blend_logistic.joblib').predict_proba(d[BASE])[:,1]
    p=calibrated(variant['calibration'],joblib.load(ART/'calibrator.joblib'),p)
    y=d.buggy.to_numpy();t=freeze['deployment_threshold']
    final=flat(y,p,t)
    pred=d[['commit_id','project','buggy']].copy();pred['probability']=p;pred['threshold']=t
    pred.to_parquet(OUT/'final_test_predictions.parquet',index=False)
    prior=pd.read_parquet(ROOT/'reports/risk_model/comparison_test_predictions.parquet').set_index('commit_id').loc[d.commit_id]
    assert np.array_equal(prior.buggy,y)
    old_thresholds=json.loads((ROOT/'models/risk_model/base_paper_comparison/frozen_models.json').read_text())['thresholds']
    names=['Logistic Regression','CatBoost V1','CatBoost +diffusion','CatBoost +history','CatBoost V2']
    # Reuse saved prior probabilities, never refit or reselect old models.
    pp={n:prior[n].to_numpy() for n in names}
    pp['Final optimized']=p; old_thresholds['Final optimized']=t
    comparison=[dict(model=n,**flat(y,v,old_thresholds[n])) for n,v in pp.items()]
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv',index=False)
    project_rows=[]
    for project in sorted(set(d.project)):
        ix=d.project.eq(project).to_numpy()
        for n,v in pp.items():
            project_rows.append(dict(project=project,model=n,rows=int(ix.sum()),**flat(y[ix],v[ix],old_thresholds[n])))
    pd.DataFrame(project_rows).to_csv(OUT/'per_project.csv',index=False)
    # Paired, project-stratified IID commit intervals; conditional on these three projects.
    rng=np.random.default_rng(20260911); samples={k:[] for k in KEYS}; diffs={n:{k:[] for k in KEYS} for n in ['Logistic Regression','CatBoost +history','CatBoost V2']}
    strata=[np.flatnonzero(d.project.eq(g)) for g in sorted(set(d.project))]
    for b in range(1000):
        ix=np.sort(np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]))
        fm=flat(y[ix],p[ix],t)
        for k in KEYS:samples[k].append(fm[k])
        for n in diffs:
            ref=flat(y[ix],pp[n][ix],old_thresholds[n])
            for k in KEYS:diffs[n][k].append(fm[k]-ref[k])
        if (b+1)%100==0: print(f'Confidence intervals {b+1}/1000',flush=True)
    ci=dict(method='Paired project-stratified IID commit percentile bootstrap, 1000 replicates, seed 20260911; pointwise 95%; conditional on fixed models and these projects, ignores within-project temporal dependence and selection uncertainty.',
            final={k:interval(v) for k,v in samples.items()},differences={n:{k:interval(v) for k,v in row.items()} for n,row in diffs.items()})
    save(OUT/'confidence_intervals.json',ci)
    save(OUT/'final_metrics.json',final)
    report(freeze,comparison,project_rows,ci)
    assert protected_hashes()==protocol['hashes_before']
    save(OUT/'verification.json',dict(protected_artifacts_unchanged=True,freeze_sha256=sha(ART/'frozen.json'),test_evaluations=1,test_rows=len(d),test_projects=sorted(set(d.project))))
    print(markdown_table(comparison,['model']+KEYS[:6]),flush=True)
    print('FINAL MULTI-METRIC OPTIMIZATION COMPLETE',flush=True)


def report(freeze,comparison,projects,ci):
    search=pd.read_csv(OUT/'validation_search.csv');variants=pd.read_csv(OUT/'blend_calibration.csv')
    final=comparison[-1]
    full_cv=json.loads((OUT/'frozen_pipeline_cv.json').read_text())
    lines=['# Final multi-metric optimization',
        'Completed 40 CatBoost configurations, three finalists with three project-aware CV folds each, two importance-pruning variants included in the 40, and 12 blend/calibration operating points. Selection used development data only. One frozen final model was scored once on the existing test partition.',
        'The existing test set had already been evaluated in previous phases. This is not a fresh independent confirmation; further research should use new held-out projects. Targets were aspirational and no scores or labels were altered.',
        '## Frozen research model',
        f"Candidate {freeze['candidate']}; {freeze['trees']} trees after early stopping. Feature subset: "+', '.join(freeze['feature_order'])+'.',
        'Transformations and full requested hyperparameters:\n```json\n'+json.dumps(freeze['spec'],indent=2)+'\n```',
        'Fixed settings: Logloss training objective, PRAUC early stopping metric, patience 60, seed 42, four CPU threads. effective_hyperparameters.json records all resolved CatBoost defaults as well as the selected search values.',
        f"Class weighting: {freeze['spec']['params'].get('class_weights',freeze['spec']['params'].get('auto_class_weights','none'))}. Selected threshold: {freeze['deployment_threshold']:.6f}. CatBoost probability weight: {1-freeze['variant']['weight']}; Logistic probability weight: {freeze['variant']['weight']}; calibration: {freeze['variant']['calibration']}.",
        'Thresholds are selected on validation predictions, using cross-fitted probabilities for calibrated variants. The final calibrator, when chosen, is fitted on all validation data; its score distribution can differ from cross-fitted predictions. No test threshold adjustment is permitted.',
        markdown_table([dict(operating_point=k,**v) for k,v in freeze['thresholds'].items() if v is not None],['operating_point','threshold','precision','recall','f1']),
        'Any absent high-precision or high-recall operating point means the .80 target was not achieved on the fixed threshold grid.',
        '## Selection protocol',
        '.25 ROC-AUC + .25 PR-AUC (average precision) + .20 F1 + .15 precision + .15 recall - .05 |precision-recall|. Threshold balance uses .4 F1 + .3 precision + .3 recall - .1 |precision-recall|. Finalist selection averages validation and CV objective, penalizes CV standard deviation, rewards capture@5/@10 and penalizes Brier. Exact formula and immutable source hashes are in the protocol.',
        '## Feature validity and distributions',
        'All 18 original columns were audited. Only nf/la/ld, nd/ns/ent, ndev/age/nuc and aexp/arexp/asexp enter the feature search. Label, fix flag, project identity, commit identity, year and author date are excluded as predictors. total_lines_changed is la+ld (total churn); adding a duplicate total_churn is unnecessary. No pre-change file size exists, so change_ratio is omitted. Seven ratios use only present values, with denominator guards: la/(ld+1), churn/max(nf,1), churn/max(nd,1), ent/max(nf,1), arexp/(aexp+1), asexp/(aexp+1), ndev/max(nf,1).',
        'Training-only skew >2 defines features eligible for log1p; raw, replacement logs and raw-plus-logs were compared. Fold medians are fitted only on their training subsets. Logs are deterministic nonnegative transforms. See training_distributions.csv and feature_audit.csv.',
        'Historical safety relies on the pre-change feature definitions in the [ApacheJIT paper](https://arxiv.org/html/2203.00101). This experiment does not remine histories to independently verify timestamp boundaries. The inherited split is project-separated, not a prospective temporal split. Dataset curation and SZZ uncertainty remain limitations.',
        '## Group-aware CV',
        full_cv['scope'],
        'Nine training projects are divided into three outer folds. A separate group-aware inner holdout selects early stopping and thresholds; outer folds do neither. Because shortlisting already used all training labels, these are post-selection stability diagnostics, not unbiased nested-CV estimates. Validation has only three projects and is dominated in size by Ignite.',
        markdown_table([dict(metric=k,**v) for k,v in full_cv['summary'].items()],['metric','mean','variance','sd']),
        '## All prior models and final test',markdown_table(comparison,['model']+KEYS),
        'PR-AUC is average precision. Capture@k gates ceil(k% of commits) and divides found positives by all positives; lift divides capture by nominal k%. Stable commit-hash order breaks ties identically across models. All prior results reuse saved predictions and original frozen thresholds.',
        '## Held-out projects, including regressions',markdown_table(projects,['project','model']+KEYS),
        '## Confidence intervals',ci['method'],
        'With only three held-out projects these intervals do not establish population-level project generalization. They can be too narrow when commits are dependent. No multiple-comparison adjustment is applied.',
        markdown_table([dict(metric=k,**v) for k,v in ci['final'].items()],['metric','low','high']),
        'Paired final-minus-prior metric differences (negative Brier differences are favorable):',
        markdown_table([dict(reference=n,metric=k,**v) for n,ms in ci['differences'].items() for k,v in ms.items()],['reference','metric','low','high']),
        '## Ablations and complete validation search',markdown_table(search.to_dict('records'),['id','name','objective','precision','recall','f1','roc_auc','pr_auc','brier_score']),
        'Feature importance, mean absolute SHAP (1,000 validation rows), and two-repeat validation permutation importance are saved separately. Two top-importance subsets (8 and 12) were tested; more features were not assumed better. Permutation diagnostics were produced after selection and did not trigger another search.',
        '## Ensemble and calibration',markdown_table(variants.to_dict('records'),['weight','calibration','eligible','objective','precision','recall','f1','roc_auc','pr_auc','brier_score']),
        ('The Logistic blend was retained under the multi-metric guard.' if freeze['variant']['weight'] else 'The Logistic blend did not warrant retention under the multi-metric guard; the final model uses CatBoost alone.'),
        ('Calibration was retained after cross-fitted validation improvement.' if freeze['variant']['calibration']!='none' else 'Neither calibration method warranted retention under the validation guard; native probabilities are retained.'),
        'LightGBM and XGBoost are not installed, so the optional second-tree ensemble was omitted. A four-feature Logistic Regression blend was evaluated. Calibration is leave-one-validation-project-out to avoid reporting in-sample calibration gains. Eligibility guards and all rejected candidates are retained.',
        '## Improvements and limitations']
    for r in comparison[:-1]:
        better=[k for k in KEYS if (final[k]<r[k] if k=='brier_score' else final[k]>r[k])]
        other=[k for k in KEYS if k not in better]
        lines.append(f"Versus {r['model']}: improved {', '.join(better) or 'none'}; did not improve {', '.join(other) or 'none'}.")
    for g in sorted(set(r['project'] for r in projects)):
        f=next(r for r in projects if r['project']==g and r['model']=='Final optimized')
        for name in ['Logistic Regression','CatBoost +history']:
            r=next(r for r in projects if r['project']==g and r['model']==name)
            worse=[k for k in KEYS if (f[k]>r[k] if k=='brier_score' else f[k]<r[k])]
            lines.append(f"{g}, regressions versus {name}: {', '.join(worse) or 'none'}.")
    lines += ['Runtime remains the existing four-feature CatBoost V1. The final artifact is research-only because historical and exact diffusion extraction are not available with verified live parity. No API, runtime model, labels, split assignments, prior reports or prior artifacts were changed.',
        'Reproduce only on an explicitly new experiment directory: run scripts/optimize_apachejit_multimetric.py --develop, review the frozen manifest, then --validate-frozen and --finalize. The script refuses existing search outputs and any repeated final evaluation. No optimization follows test scoring.',
        'FINAL MULTI-METRIC OPTIMIZATION COMPLETE']
    (OUT/'final_report.md').write_text('\n\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--develop',action='store_true');group.add_argument('--finalize',action='store_true');group.add_argument('--validate-frozen',action='store_true')
    args=parser.parse_args()
    develop() if args.develop else (validate_frozen() if args.validate_frozen else finalize())
