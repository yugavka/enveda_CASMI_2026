"""Learned ranker for generated class-3 candidates (LightGBM LambdaRank).

Trains on the candidate rows that class3/bench.py dumps for one query set and evaluates on a disjoint set against
the hand-made neural + fragment + parent ranking. A query is one (structure, threshold) pair; queries without the
truth among their candidates count as reciprocal rank 0 for every method.

    .venv/bin/python -I class3/ranker.py --train class3/results_rank_train --eval class3/results_rank_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

NUMERIC = ['frag', 'neural', 'parent_sim', 'parent_rank', 'z_frag', 'z_neural', 'z_parent', 'rank_frag', 'rank_neural',
           'support', 'n_steps', 'tan_parent', 'heavy', 'heavy_parent', 'n_cands']
CATEGORICAL = ['rule1', 'rule2']
PARAMS = dict(objective='lambdarank', metric='ndcg', eval_at=[1, 5, 25], learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0)


def load(d):
    c = pd.read_parquet(Path(d) / 'candidates.parquet')
    q = pd.read_csv(Path(d) / 'per_query.csv')
    q['qid'] = q.ik + '|' + q.thr.astype(str)
    c['qid'] = c.ik + '|' + c.thr.astype(str)
    return c, q


def matrix(c, cats):
    x = c[NUMERIC + CATEGORICAL].copy()
    for col in CATEGORICAL:
        x[col] = pd.Categorical(x[col], categories=cats[col])
    return x


def rr25(c, score):
    """Reciprocal rank (tie block middle, 0 beyond 25) of the label in each qid, under `score`."""
    c = c.assign(s=score)
    out = {}
    for qid, g in c.groupby('qid', sort=False):
        if not g.label.any():
            continue
        s, t = g.s.to_numpy(), g.s[g.label == 1].iloc[0]
        r = 1 + (s > t).sum() + ((s == t).sum() - 1) / 2
        out[qid] = 1.0 / r if r <= 25 else 0.0
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='class3/results_rank_train')
    ap.add_argument('--eval', default='class3/results_rank_eval')
    ap.add_argument('--out', default='class3/ranker')
    ap.add_argument('--rounds', type=int, default=2000)
    a = ap.parse_args()
    tr, _ = load(a.train)
    ev, evq = load(a.eval)
    cats = {col: sorted(set(tr[col]) | set(ev[col])) for col in CATEGORICAL}

    # train only on queries whose truth was generated; early-stop on 20% of the training structures
    tr = tr[tr.groupby('qid').label.transform('max') == 1].sort_values('qid', kind='stable')
    iks = np.array(sorted(tr.ik.unique()))
    rng = np.random.default_rng(0)
    inner = set(rng.choice(iks, size=len(iks) // 5, replace=False))
    fit, val = tr[~tr.ik.isin(inner)], tr[tr.ik.isin(inner)]
    group = lambda d: d.groupby('qid', sort=False).size().to_numpy()
    dfit = lgb.Dataset(matrix(fit, cats), fit.label, group=group(fit), categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(matrix(val, cats), val.label, group=group(val), categorical_feature=CATEGORICAL, reference=dfit)
    model = lgb.train(PARAMS, dfit, a.rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    print(f'train queries with truth: {tr.qid.nunique()} ({fit.qid.nunique()} fit / {val.qid.nunique()} early-stop), '
          f'best iteration {model.best_iteration}')

    # evaluate: every eval query counts; no truth generated -> 0
    scores = {'frag': ev.frag, 'neural+frag+parent': ev.z_neural + ev.z_frag + 0.5 * ev.z_parent,
              'ranker': model.predict(matrix(ev, cats), num_iteration=model.best_iteration)}
    base = pd.Series(0.0, index=evq.qid)
    res = pd.DataFrame({k: base.add(rr25(ev, v), fill_value=0).reindex(evq.qid).fillna(0.0) for k, v in scores.items()})
    res['thr'] = evq.set_index('qid').thr.reindex(res.index).to_numpy()
    res['ik'] = evq.set_index('qid').ik.reindex(res.index).to_numpy()
    summary = res.groupby('thr')[list(scores)].mean()
    print('MRR@25 on the eval queries (all queries; not generated = 0):')
    print(summary.round(4).to_string())
    top1 = {k: rr25(ev, v).pipe(lambda s: float((s == 1).mean())) for k, v in scores.items()}
    print('rank 1 when generated:', {k: round(v, 3) for k, v in top1.items()})

    # paired bootstrap over structures (all thresholds of a structure resampled together)
    per_ik = res.groupby('ik')[list(scores)].mean()
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(per_ik), (4000, len(per_ik)))
    for other in ('neural+frag+parent', 'frag'):
        d = (per_ik['ranker'] - per_ik[other]).to_numpy()
        m = d[idx].mean(1)
        print(f'ranker - {other}: {d.mean():+.4f} [{np.quantile(m, .025):+.4f}, {np.quantile(m, .975):+.4f}]')

    imp = pd.Series(model.feature_importance('gain'), index=NUMERIC + CATEGORICAL).sort_values(ascending=False)
    print('feature gain:', (imp / imp.sum()).round(3).to_dict())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / 'ranker.txt'), num_iteration=model.best_iteration)
    summary.to_csv(out / 'eval_summary.csv')
    (out / 'meta.json').write_text(json.dumps(dict(numeric=NUMERIC, categorical=CATEGORICAL, categories=cats,
                                                   params=PARAMS, best_iteration=model.best_iteration), indent=2))


if __name__ == '__main__':
    main()
