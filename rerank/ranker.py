"""Learned isomer reranker (LightGBM LambdaRank) on the rows rerank/bench.py dumps.

Splits the query structures 75/25 (by a hash of the InChIKey block), trains on the first part (early stop on a fifth
of it) and reports MRR@25 on the rest against the base order (z_neural + z_frag). Queries whose truth missed the
shortlist count as 0 for every method.

    .venv/bin/python -I rerank/ranker.py --bench rerank/results --out rerank/model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / "class3")]
from features import NUMERIC  # noqa: E402

PARAMS = dict(objective='lambdarank', metric='ndcg', eval_at=[1, 3, 10], learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0)


def part(ik, buckets=4):
    return int(hashlib.md5(ik.encode()).hexdigest(), 16) % buckets


def rr25(c, score):
    """Reciprocal rank (tie block middle, 0 beyond 25) of the label in each qid that has one."""
    c = c.assign(s=np.asarray(score, float))
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
    ap.add_argument('--bench', default='rerank/results')
    ap.add_argument('--out', default='rerank/model')
    ap.add_argument('--drop', default='', help='comma-separated features to leave out')
    a = ap.parse_args()
    c = pd.read_parquet(Path(a.bench) / 'candidates.parquet')
    feats = [f for f in NUMERIC if f not in set(a.drop.split(','))]
    c['part'] = c.ik.map(part)
    tr, ev = c[c.part != 0], c[c.part == 0]
    qids = c[['qid', 'ik', 'thr', 'part']].drop_duplicates('qid')
    evq = qids[qids.part == 0]

    tr = tr[tr.groupby('qid').label.transform('max') == 1].sort_values('qid', kind='stable')
    inner = tr.ik.map(lambda k: part(k, 5) == 1)
    fit, val = tr[~inner], tr[inner]
    group = lambda d: d.groupby('qid', sort=False).size().to_numpy()
    model = lgb.train(PARAMS, lgb.Dataset(fit[feats], fit.label, group=group(fit)), 3000,
                      valid_sets=[lgb.Dataset(val[feats], val.label, group=group(val))],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    print(f'train queries {tr.qid.nunique()} (fit {fit.qid.nunique()} / stop {val.qid.nunique()}), '
          f'best iteration {model.best_iteration}')

    scores = {'base': -ev.base_rank, 'neural': ev.neural, 'frag': ev.frag, 'support': ev.sup_wsum,
              'ranker': model.predict(ev[feats], num_iteration=model.best_iteration)}
    res = pd.DataFrame({k: pd.Series(0.0, index=evq.qid).add(rr25(ev, v), fill_value=0).reindex(evq.qid).fillna(0.0)
                        for k, v in scores.items()})
    res['thr'] = evq.thr.to_numpy()
    res['ik'] = evq.ik.to_numpy()
    print(f'MRR@25 on {evq.ik.nunique()} eval structures:')
    print(res.groupby('thr')[list(scores)].mean().round(4).to_string())
    print('top1:', {k: round(float((res[k] == 1).mean()), 3) for k in scores})
    per_ik = res.groupby('ik')[list(scores)].mean()
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(per_ik), (4000, len(per_ik)))
    d = (per_ik.ranker - per_ik.base).to_numpy()
    m = d[idx].mean(1)
    print(f'ranker - base: {d.mean():+.4f} [{np.quantile(m, .025):+.4f}, {np.quantile(m, .975):+.4f}]')
    imp = pd.Series(model.feature_importance('gain'), index=feats).sort_values(ascending=False)
    print('gain:', (imp / imp.sum()).round(3).to_dict())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / 'ranker.txt'), num_iteration=model.best_iteration)
    (out / 'meta.json').write_text(json.dumps(dict(numeric=feats, params=PARAMS,
                                                   best_iteration=model.best_iteration), indent=2))
    res.to_csv(out / 'eval_rr.csv')


if __name__ == '__main__':
    main()
