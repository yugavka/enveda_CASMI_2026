"""Same-formula isomer reranking bench: can a learned ranker reorder a strong shortlist better than its own order?

Simulates a molecule the pipeline cannot confirm by library match: the truth's spectra are removed from the spectral
library (THR < 1 also removes its close analogs, Morgan Tanimoto >= THR). Candidates are the truth's same-formula
isomers among the training structures (at most CAP). They are scored with the fragment scorer and the neural scorer
(validation checkpoint; the queries are its validation split), the BASE order (z_neural + z_frag) keeps a shortlist of
SHORT and every shortlisted candidate gets ranker features, including support from the query's spectral neighbours:
how similar the candidate is to library structures whose spectra resemble the query (its own spectra excluded).

    .venv/bin/python -I rerank/bench.py --n 2000 --out rerank/results
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
sys.path[:0] = [str(HERE), str(HERE.parent / 'class3'), str(HERE.parent / 'frag')]
import candidates as C  # noqa: E402
import library as L  # noqa: E402
import features as Fe  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('class3_bench', HERE.parent / 'class3' / 'bench.py')
_c3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c3)
load_queries = _c3.load_queries

RDLogger.DisableLog('rdApp.*')
THRESHOLDS = (1.0, 0.85)
CAP = 200
SHORT = 25
_SCORER = None


def _init_worker(model_dir, checkpoint):
    global _SCORER
    import torch
    torch.set_num_threads(1)
    from neural import Scorer
    _SCORER = Scorer(model_dir, checkpoint=checkpoint)


def _work(args):
    q, cands, neighbours = args
    truth = q['ik']
    keys = [k for k, _ in cands]
    smiles = [s for _, s in cands]
    feats, base_rank = Fe.features(q, keys, smiles, neighbours, _SCORER, short=SHORT)
    rows = []
    for thr, f in feats.items():
        for r in f:
            r.update(ik=truth, thr=thr, qid=f'{truth}|{thr}', label=int(r['cand'] == truth))
        rows += f
    rank = base_rank.get(truth, np.inf)
    return dict(ik=truth, lib=q.get('lib'), n_iso=len(keys), base_rank=rank), rows


def isomer_pool(data):
    t = ds.dataset(data).to_table(columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).to_pandas()
    t = t.drop_duplicates('inchikey14').set_index('inchikey14')
    return t, {f: list(g.index) for f, g in t.groupby('molecular_formula', sort=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train.parquet')
    ap.add_argument('--cache', default='data/cache')
    ap.add_argument('--out', default='rerank/results')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='data/models/sanpier/casmi_spectrum_model_v1')
    ap.add_argument('--checkpoint', default='best_validation_model.pt')
    ap.add_argument('--k', type=int, default=Fe.K_NEIGHBOURS)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t0 = time.time()
    import frag_rescore
    assert frag_rescore.fragment_levels('CCOC(=O)c1ccccc1O') is not None, 'fragment scorer broken (stale numba cache?)'
    meta, frag_m, nl_m = L.load(Path(a.cache))
    structs = meta.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles
    mols = [Chem.MolFromSmiles(s) for s in structs]
    ok = np.array([m is not None for m in mols])
    structs = structs[ok]
    fps = [C.FPG.GetFingerprint(m) for m in mols if m is not None]
    info, by_formula = isomer_pool(a.data)
    queries = load_queries(a.data, 'val_np', a.n, a.seed, a.model, (), np_first=False)
    print(f'library {len(structs):,} structures, {len(queries)} queries ({time.time() - t0:.0f}s)', flush=True)

    rng = np.random.default_rng(a.seed)
    jobs = []
    for n_done, q in enumerate(queries, 1):
        if n_done % 250 == 0:
            print(f'jobs {n_done}/{len(queries)} ({time.time() - t0:.0f}s)', flush=True)
        iso = by_formula.get(info.molecular_formula.get(q['ik']), [q['ik']])
        if len(iso) > CAP:
            rest = [k for k in iso if k != q['ik']]
            iso = [q['ik']] + list(rng.choice(rest, CAP - 1, replace=False))
        cands = [(k, info.normalized_smiles[k]) for k in iso]
        sims = C.neighbour_scores(q, meta, frag_m, nl_m).reindex(structs.index).fillna(0.0)
        tan = np.array(DataStructs.BulkTanimotoSimilarity(C.FPG.GetFingerprint(Chem.MolFromSmiles(q['smiles'])), fps))
        neighbours = {}
        for thr in THRESHOLDS:
            keep = (tan < thr) & (structs.index != q['ik'])
            top = sims[keep].nlargest(a.k + 5)  # spare ones: a candidate's own entry is skipped in its support
            neighbours[thr] = [(i, structs[i], float(v)) for i, v in top.items()]
        jobs.append((q, cands, neighbours))
    print(f'jobs ready ({time.time() - t0:.0f}s)', flush=True)

    rows, feats = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(a.model, a.checkpoint)) as pool:
        for i, (row, f) in enumerate(pool.map(_work, jobs, chunksize=2), 1):
            rows.append(row)
            feats += f
            if i % 200 == 0:
                print(f'{i}/{len(jobs)} ({time.time() - t0:.0f}s)', flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_query.csv', index=False)
    pd.DataFrame(feats).to_parquet(out / 'candidates.parquet')
    (out / 'config.json').write_text(json.dumps(dict(vars(a), thresholds=THRESHOLDS, cap=CAP, short=SHORT), indent=2))
    r = df.base_rank
    print(f'isomers median {df.n_iso.median():.0f}; truth in shortlist {np.mean(r <= SHORT):.3f}; '
          f'base MRR@25 {np.mean(np.where(r <= 25, 1 / r, 0)):.4f}; base top1 {np.mean(r == 1):.3f}')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
