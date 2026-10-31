"""Class-3 insertion stage for the Kaggle notebook (runs after the final merge layer).

For each molecule whose best library match is below `gate` (the pipeline's leader is not library-confirmed), find
its spectral neighbours in a library built from train.parquet, generate analogs of the query mass, rank them with
the learned LambdaRank model and insert the best `n_insert` new structures at rank `position` (1-based); ranks
above it are kept, the rest shift down and the list is cut at 25.

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
import library as L

DEFAULTS = dict(gate=0.87, position=3, n_insert=3, k=30, steps=2, max_spectra=6, seconds=2400, workers=0,
                checkpoint='final_model.pt', limit=0)
_SCORER = None


def _init(bundle, checkpoint):
    global _SCORER
    import torch
    torch.set_num_threads(1)
    from neural import Scorer
    _SCORER = Scorer(bundle, checkpoint=checkpoint)


def _one(args):
    mid, q, neighbours, steps = args
    keys, _, feats = C.build(q, neighbours, steps, _SCORER, qid=mid)
    return mid, feats


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
    gated = [m for m in sub.molecule_id if float(lib_max.get(str(m), 0.0)) < c['gate']]
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
    rep['library_seconds'] = round(time.time() - t0, 1)

    jobs = []
    for mid in gated:
        q = queries.get(str(mid))
        if q is None:
            continue
        top = C.neighbour_scores(q, meta_lib, frag_m, nl_m).reindex(structs.index).fillna(0.0).nlargest(c['k'])
        jobs.append((str(mid), q, [(structs[i], float(v)) for i, v in top.items()], c['steps']))
    rep['neighbour_seconds'] = round(time.time() - t0, 1)

    model = lgb.Booster(model_file=str(ranker_dir / 'ranker.txt'))
    meta = json.loads((ranker_dir / 'meta.json').read_text())
    workers = c['workers'] or max(1, (os.cpu_count() or 2))
    ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else None
    picks, timed_out = {}, 0
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(str(bundle), c['checkpoint'])) as pool:
        futures = [pool.submit(_one, j) for j in jobs]
        for f in futures:
            left = c['seconds'] - (time.time() - t0)
            try:
                mid, feats = f.result(timeout=max(left, 1))
            except Exception:  # timeout or a failed molecule: that molecule keeps its list
                timed_out += 1
                continue
            if feats:
                s = rank_features(feats, model, meta)
                order = np.argsort(-s, kind='stable')
                picks[mid] = [feats[i]['smiles'] for i in order]
        for f in futures:
            f.cancel()

    changed, inserted = 0, 0
    rows = []
    for mid, smi in zip(sub.molecule_id, sub.smiles):
        old = smi.split(';')
        text = smi  # rows we do not change pass through byte for byte
        if mid in picks:
            have = {C.k14(s) for s in old}
            add = [s for s in picks[mid] if C.k14(s) not in have][:c['n_insert']]
            if add:
                new = insert(old, add, c['position'])
                assert 1 <= len(new) <= 25 and len(set(new)) == len(new) and all(new), mid
                inserted += len(set(new) - set(old))
                changed += 1
                text = ';'.join(new)
        rows.append((mid, text))
    out = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    assert out.molecule_id.tolist() == sub.molecule_id.tolist()
    tmp = root / 'submission.class3.tmp'
    out.to_csv(tmp, index=False)
    os.replace(tmp, root / 'submission.csv')
    return dict(rep, status='applied', jobs=len(jobs), with_candidates=len(picks), changed_rows=int(changed),
                inserted=int(inserted), timed_out=timed_out, seconds=round(time.time() - t0, 1))
