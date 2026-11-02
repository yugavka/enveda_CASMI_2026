"""Class-3 insertion stage for the Kaggle notebook (runs after the final merge layer).

For each molecule whose best library match is below `gate` (the pipeline's leader is not library-confirmed), find
its spectral neighbours in a library built from train.parquet, generate analogs of the query mass, rank them with
the learned LambdaRank model and insert the best `n_insert` new structures at rank `position` (1-based); ranks
above it are kept, the rest shift down and the list is cut at 25. When the ranker's top-1 margin over the runner-up
is at least `confident_margin`, the top pick goes to `confident_position` instead (eval: P(top1) ~0.40 at margin 0.8).

`rerank` ('all' | 'freeze1' | None): before the insertion, the same gated lists are reordered by the isomer reranker
(rerank/; trained on held-out molecules whose spectra were removed from the library); 'freeze1' keeps the leader.

`variants` {name: overrides of gate/position/n_insert/confident_*} also writes submission_<name>.csv from the same
candidates, so one notebook run yields several LB probes.

    run(root, comp, bundle, lib_max, cfg) -> report dict   (submission.csv in `root` is rewritten in place)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import candidates as C
import features as RF
import library as L

DEFAULTS = dict(gate=0.87, position=3, n_insert=3, confident_margin=None, confident_position=1, k=30, steps=2,
                max_spectra=6, seconds=2400, workers=0, checkpoint='final_model.pt', limit=0, variants={},
                rerank=None)
PLACEMENT = ('gate', 'position', 'n_insert', 'confident_margin', 'confident_position')
_SCORER = None


def _init(bundle, checkpoint):
    global _SCORER
    import torch
    torch.set_num_threads(1)
    from neural import Scorer
    _SCORER = Scorer(bundle, checkpoint=checkpoint)


def _one(args):
    mid, q, neighbours, steps, rr = args
    keys, _, feats = C.build(q, neighbours, steps, _SCORER, qid=mid)
    rows = []
    if rr is not None:
        smiles, rr_neighbours = rr
        rows = RF.features(q, [C.k14(s) for s in smiles], smiles, {1.0: rr_neighbours}, _SCORER,
                           short=len(smiles), order=[C.k14(s) for s in smiles])[0][1.0]
    return mid, feats, rows


def rerank_list(old, rows, model, meta, mode):
    """`old` reordered by the reranker score over one entry per InChIKey14 block (the first); the other entries of
    a block follow their representative, entries without a valid structure go to the end."""
    if not rows:
        return old
    s = model.predict(pd.DataFrame(rows)[meta['numeric']])
    keys = [rows[i]['cand'] for i in np.argsort(-s, kind='stable')]
    if mode == 'freeze1' and C.k14(old[0]) in keys:
        keys = [C.k14(old[0])] + [k for k in keys if k != C.k14(old[0])]
    block = {}
    for x in old:
        block.setdefault(C.k14(x), []).append(x)
    out = [x for k in keys for x in block.get(k, [])]
    return out + [x for x in old if x not in set(out)]


def queries_from_test(test_path, max_spectra):
    t = pd.read_parquet(test_path)
    t = t[t.adduct.isin(list(L.ADDUCTS)) & (t.ms2_mzs.map(len) >= 5)]
    out = {}
    for mid, g in t.groupby('molecule_id', sort=False):
        g = g.head(max_spectra)
        spectra = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float),
                        adduct=r.adduct, prec=float(r.precursor_mz), ce=r.collision_energy_ev,
                        mode=1 if str(r.ionization_mode).lower() == 'positive' else -1)
                   for r in g.itertuples(index=False)]
        out[str(mid)] = dict(spectra=spectra,
                             mass=float(np.median([L.neutral_mass(s['prec'], s['adduct']) for s in spectra])))
    return out


def rank_features(feats, model, meta):
    x = pd.DataFrame(feats)[meta['numeric'] + meta['categorical']]
    for col in meta['categorical']:
        x[col] = pd.Categorical(x[col], categories=meta['categories'][col])
    return model.predict(x)


def insert(old, new, position, limit=25):
    """old[:position-1] + new + the rest of old, without duplicates, at most `limit` entries."""
    head, tail = old[:position - 1], old[position - 1:]
    out = []
    for s in head + new + tail:
        if s and s not in out:
            out.append(s)
    return out[:limit]


def run(root, comp, bundle, lib_max, cfg=None, ranker_dir=None, cache=None):
    import lightgbm as lgb
    c = dict(DEFAULTS, **(cfg or {}))
    t0 = time.time()
    root, comp = Path(root), Path(comp)
    ranker_dir = Path(ranker_dir or Path(__file__).resolve().parent / 'ranker')
    sub = pd.read_csv(root / 'submission.csv', dtype=str, keep_default_na=False)
    max_gate = max([c['gate']] + [v.get('gate', c['gate']) for v in c['variants'].values()])
    gated = [m for m in sub.molecule_id if float(lib_max.get(str(m), 0.0)) < max_gate]
    if c['limit']:
        gated = gated[:c['limit']]
    rep = dict(status='started', cfg=c, molecules=len(sub), gated=len(gated))
    if not gated:
        return dict(rep, status='nothing_gated')

    cache = Path(cache or root / 'class3_cache')
    if not (cache / 'class3_frag.npz').is_file():
        L.build(comp / 'train.parquet', cache)
    meta_lib, frag_m, nl_m = L.load(cache)
    structs = meta_lib.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles
    queries = queries_from_test(comp / 'test.parquet', c['max_spectra'])
    sub_lists = {str(m): x.split(';') for m, x in zip(sub.molecule_id, sub.smiles)}
    rep['library_seconds'] = round(time.time() - t0, 1)

    jobs = []
    for mid in gated:
        q = queries.get(str(mid))
        if q is None:
            continue
        top = C.neighbour_scores(q, meta_lib, frag_m, nl_m).reindex(structs.index).fillna(0.0).nlargest(c['k'] + 5)
        rr = None
        old = sub_lists[str(mid)]
        reps = {}  # first entry of each InChIKey14 block, in list order
        for x in old:
            reps.setdefault(C.k14(x), x)
        reps.pop(None, None)
        reps = list(reps.items())
        if c['rerank'] and len(reps) > 1:
            rr = ([x for _, x in reps], [(i, structs[i], float(v)) for i, v in top.items()])
        top = top.head(c['k'])
        jobs.append((str(mid), q, [(structs[i], float(v)) for i, v in top.items()], c['steps'], rr))
    rep['neighbour_seconds'] = round(time.time() - t0, 1)

    model = lgb.Booster(model_file=str(ranker_dir / 'ranker.txt'))
    meta = json.loads((ranker_dir / 'meta.json').read_text())
    rr_dir = ranker_dir.parent / 'rerank'
    if c['rerank']:
        rr_model = lgb.Booster(model_file=str(rr_dir / 'ranker.txt'))
        rr_meta = json.loads((rr_dir / 'meta.json').read_text())
    reranked = {}
    workers = c['workers'] or max(1, (os.cpu_count() or 2))
    ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else None
    picks, timed_out = {}, 0
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(str(bundle), c['checkpoint'])) as pool:
        futures = [pool.submit(_one, j) for j in jobs]
        for f in futures:
            left = c['seconds'] - (time.time() - t0)
            try:
                mid, feats, rr_rows = f.result(timeout=max(left, 1))
            except Exception:  # timeout or a failed molecule: that molecule keeps its list
                timed_out += 1
                continue
            if rr_rows and float(lib_max.get(mid, 0.0)) < c['gate']:
                new = rerank_list(sub_lists[mid], rr_rows, rr_model, rr_meta, c['rerank'])
                if new != sub_lists[mid]:
                    reranked[mid] = new
            if feats:
                s = rank_features(feats, model, meta)
                order = np.argsort(-s, kind='stable')
                picks[mid] = ([feats[i]['smiles'] for i in order], s[order])
        for f in futures:
            f.cancel()

    rep.update(jobs=len(jobs), with_candidates=len(picks), timed_out=timed_out, reranked=len(reranked),
               new_leader=sum(v[0] != sub_lists[m][0] for m, v in reranked.items()))
    main = {k: c[k] for k in PLACEMENT}
    out, stats = place(sub, picks, lib_max, main, reranked)
    _write(out, root / 'submission.csv')
    rep['variants'] = {}
    for name, v in c['variants'].items():
        vout, vstats = place(sub, picks, lib_max, dict(main, **v), reranked)
        _write(vout, root / f'submission_{name}.csv')
        rep['variants'][name] = vstats
    return dict(rep, status='applied', **stats, seconds=round(time.time() - t0, 1))


def place(sub, picks, lib_max, p, reranked=None):
    """Submission with the (reranked) lists and the ranked picks inserted under placement `p`; unchanged rows pass
    through byte for byte."""
    reranked = reranked or {}
    changed, inserted, confident = 0, 0, 0
    rows = []
    for mid, smi in zip(sub.molecule_id, sub.smiles):
        text = ';'.join(reranked[mid]) if mid in reranked else smi
        if mid in picks and float(lib_max.get(str(mid), 0.0)) < p['gate']:
            old = text.split(';')
            have = {C.k14(x) for x in old}
            ranked, scores = picks[mid]
            keep = [i for i, x in enumerate(ranked) if C.k14(x) not in have][:p['n_insert']]
            add = [ranked[i] for i in keep]
            if add:
                margin = scores[0] - scores[1] if len(scores) > 1 else np.inf
                if p['confident_margin'] is not None and keep[0] == 0 and margin >= p['confident_margin']:
                    new = insert(insert(old, add[1:], p['position']), add[:1], p['confident_position'])
                    confident += 1
                else:
                    new = insert(old, add, p['position'])
                assert 1 <= len(new) <= 25 and len(set(new)) == len(new) and all(new), mid
                inserted += len(set(new) - set(old))
                changed += 1
                text = ';'.join(new)
        rows.append((mid, text))
    out = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    assert out.molecule_id.tolist() == sub.molecule_id.tolist()
    return out, dict(changed_rows=int(changed), inserted=int(inserted), confident=int(confident), placement=p)


def _write(df, path):
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
