"""Class-3 simulation bench: can analog generation recover structures that are in no database?

For every query structure (enveda-np-examples: timsTOF natural products) we remove from the spectral library the
structure itself and every structure with Morgan Tanimoto >= THR to it (THR = 1.0 keeps all analogs, 0.85 and 0.7
remove close ones), find the K spectrally nearest remaining structures (fragment + neutral-loss cosine), apply
mass-delta biotransformations (class3/transforms.py) and rank the generated candidates. Reports how often the truth
is generated at all, and its rank under a few simple rankings.

    .venv/bin/python -I class3/bench.py --out class3/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rdkit import Chem, DataStructs, RDLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'frag'))
import candidates as C  # noqa: E402
import library as L  # noqa: E402
import transforms as T  # noqa: E402

RDLogger.DisableLog('rdApp.*')
THRESHOLDS = (1.0, 0.85, 0.7)
K_NEIGHBOURS = 30
MAX_QUERY_SPECTRA = 6
FPG = C.FPG
NP_LIBS = ['enveda-np-examples', 'gnps', 'riken', 'mona', 'msdial', 'massbank']
_SCORER = None


def _init_worker(model_dir, checkpoint):
    """Per-process neural scorer (CPU, one thread) and a working fragment scorer."""
    global _SCORER
    import torch
    torch.set_num_threads(1)
    if model_dir:
        from neural import Scorer  # our re-implementation; only weights are read from the bundle
        _SCORER = Scorer(model_dir, checkpoint=checkpoint)


def load_queries(data, source='np', n=0, seed=0, model_dir=None, exclude=(), np_first=True):
    """Query molecules. 'np': all enveda-np-examples structures. 'val_np': structures the neural model held out
    (its validation split) from the natural-product libraries, timsTOF ones first, `n` sampled."""
    libs = ['enveda-np-examples'] if source == 'np' else NP_LIBS
    cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'adduct', 'ionization_mode', 'precursor_mz', 'ms2_mzs',
            'ms2_normalized_intensities', 'collision_energy_ev']
    t = ds.dataset(data).to_table(columns=cols, filter=ds.field('ingest_lib').isin(libs)).to_pandas()
    t = t[t.adduct.isin(list(L.ADDUCTS)) & (t.ms2_mzs.map(len) >= 5)]
    if source == 'val_np':
        mol = pd.read_parquet(Path(model_dir) / 'molecules.parquet', columns=['key', 'split'])
        val = set(mol.key[mol.split == 'validation'])
        t = t[t.inchikey14.isin(val) & ~t.inchikey14.isin(set(exclude))]
        iks = sorted(t.inchikey14.unique())
        np_first = set(t.inchikey14[t.ingest_lib == 'enveda-np-examples']) if np_first else set()
        rng = np.random.default_rng(seed)
        rest = [k for k in iks if k not in np_first]
        pick = sorted(np_first) + list(rng.permutation(rest))
        t = t[t.inchikey14.isin(set(pick[:n] if n else pick))]
    out = []
    for ik, g in t.groupby('inchikey14', sort=True):
        g = g.sort_values('ingest_lib', key=lambda s: s != 'enveda-np-examples', kind='stable').head(MAX_QUERY_SPECTRA)
        spectra = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float),
                        adduct=r.adduct, prec=float(r.precursor_mz), ce=r.collision_energy_ev,
                        mode=1 if r.ionization_mode == 'positive' else -1) for r in g.itertuples(index=False)]
        mass = float(np.median([L.neutral_mass(s['prec'], s['adduct']) for s in spectra]))
        out.append(dict(ik=ik, smiles=g.normalized_smiles.iloc[0], spectra=spectra, mass=mass,
                        lib=g.ingest_lib.iloc[0]))
    return out


def _work(args):
    """Generate from the neighbours of one query under one threshold and rank the candidates."""
    q, thr, neighbours, steps = args
    truth = q['ik']
    keys, raw, feats = C.build(q, neighbours, steps, _SCORER, qid=f"{truth}|{thr}", truth=truth)
    for f in feats:
        f.update(ik=truth, thr=thr)
    row = dict(ik=truth, lib=q.get('lib'), thr=thr, n_neighbours=len(neighbours), n_cands=len(keys),
               best_neighbour_sim=neighbours[0][1] if neighbours else 0.0, generated=truth in keys)
    if not keys:
        return row, []
    fs, psim, prank = raw['frag'], raw['parent_sim'], raw['parent_rank']
    zf, zn, zp = raw['z_frag'], raw['z_neural'], raw['z_parent']
    rankings = {'frag': fs, 'parent': psim - 1e-3 * prank + 1e-6 * fs, 'frag+parent': fs + 0.5 * psim}
    if _SCORER is not None:
        rankings.update({'neural': zn, 'neural+frag': zn + zf, 'neural+frag+parent': zn + zf + 0.5 * zp,
                         'neural2+frag': 2 * zn + zf})
    for name, score in rankings.items():
        if truth in keys:
            i = keys.index(truth)
            ties = int((score == score[i]).sum()) - 1
            r = 1 + int((score > score[i]).sum()) + ties / 2  # the middle of its tie block
            row[f'rank_{name}'] = r
            row[f'rr25_{name}'] = 1.0 / r if r <= 25 else 0.0
            row[f'ties_{name}'] = ties
        else:
            row[f'rank_{name}'] = np.nan
            row[f'rr25_{name}'] = 0.0
    return row, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train.parquet')
    ap.add_argument('--cache', default='data/cache')
    ap.add_argument('--out', default='class3/results')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--source', default='np', choices=['np', 'val_np'])
    ap.add_argument('--n', type=int, default=300, help='queries for --source val_np')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='data/models/sanpier/casmi_spectrum_model_v1',
                    help="neural model bundle: config.json + checkpoint ('' to skip the neural ranking)")
    ap.add_argument('--checkpoint', default='best_validation_model.pt')
    ap.add_argument('--k', type=int, default=K_NEIGHBOURS, help='spectral neighbours per query')
    ap.add_argument('--steps', type=int, default=2, help='max transformations chained')
    ap.add_argument('--exclude', default='', help='per_query.csv whose query structures are skipped (disjoint sets)')
    ap.add_argument('--no-np-first', action='store_true', help='do not force enveda-np-examples queries in')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t0 = time.time()
    import frag_rescore
    assert frag_rescore.fragment_levels('CCOC(=O)c1ccccc1O') is not None, 'fragment scorer broken (stale numba cache?)'
    meta, frag_m, nl_m = L.load(Path(a.cache))
    structs = meta.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles
    fps = [FPG.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None for s in structs]
    ok = [f is not None for f in fps]
    structs, fps = structs[ok], [f for f in fps if f is not None]
    exclude = set(pd.read_csv(a.exclude).ik) if a.exclude else set()
    queries = load_queries(a.data, a.source, a.n, a.seed, a.model, exclude, not a.no_np_first)
    if a.limit:
        queries = queries[:a.limit]
    print(f'library {len(structs):,} structures, {len(queries)} queries ({time.time() - t0:.0f}s)', flush=True)

    jobs = []
    for q in queries:
        sims = C.neighbour_scores(q, meta, frag_m, nl_m).reindex(structs.index).fillna(0.0)
        tan = np.array(DataStructs.BulkTanimotoSimilarity(FPG.GetFingerprint(Chem.MolFromSmiles(q['smiles'])), fps))
        for thr in THRESHOLDS:
            keep = (tan < thr) & (structs.index != q['ik'])
            top = sims[keep].nlargest(a.k)
            jobs.append((q, thr, [(structs[i], float(v)) for i, v in top.items()], a.steps))
    print(f'neighbours ready: {len(jobs)} jobs ({time.time() - t0:.0f}s)', flush=True)

    rows, feats = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(a.model, a.checkpoint)) as pool:
        for i, (row, f) in enumerate(pool.map(_work, jobs, chunksize=2), 1):
            rows.append(row)
            feats += f
            if i % 150 == 0:
                print(f'{i}/{len(jobs)} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    rr = [c for c in df.columns if c.startswith('rr25_')]
    df[rr] = df[rr].fillna(0.0)  # no candidates at all = reciprocal rank 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'per_query.csv', index=False)
    pd.DataFrame(feats).to_parquet(out / 'candidates.parquet')  # ranker training/eval rows
    names = [c[5:] for c in rr]
    agg = df.groupby('thr').agg(n=('ik', 'size'), generated=('generated', 'mean'), median_cands=('n_cands', 'median'),
                                **{f'mrr25_{k}': (f'rr25_{k}', 'mean') for k in names},
                                **{f'top1_{k}': (f'rank_{k}', lambda r: float((r <= 1).mean())) for k in names})
    agg.to_csv(out / 'summary.csv')
    (out / 'config.json').write_text(json.dumps(dict(vars(a), thresholds=THRESHOLDS,
                                                     cap=C.CAP_PER_NEIGHBOUR, rules=[r[0] for r in T.RULES]), indent=2))
    pd.set_option('display.width', 200)
    print(agg.round(3).to_string())
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
