"""Reranker features for one query's candidate shortlist; shared by rerank/bench.py and the notebook stage.

features(q, keys, smiles, neighbours, scorer, short, order=None) -> ({thr: [row, ...]}, {key: base rank})

neighbours: {thr: [(inchikey14, smiles, spectral similarity), ...] best first}. order: candidate keys in the base
order (the pipeline's list in the notebook); None ranks by z_neural + z_frag over all candidates (the bench).
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs

import candidates as C

K_NEIGHBOURS = 30
NUMERIC = ['frag', 'neural', 'z_frag', 'z_neural', 'rank_frag', 'rank_neural', 'base_rank', 'n_short', 'heavy',
           'sup_max', 'sup_top1', 'sup_mean5', 'sup_wsum', 'z_sup', 'best_nb_sim']


def _rank(v):
    return (-np.nan_to_num(np.asarray(v, float), nan=-1e9)).argsort(kind='stable').argsort() + 1


def features(q, keys, smiles, neighbours, scorer, short=25, order=None):
    import frag_rescore as frag
    if order is None:
        fs = np.array([np.nan if v is None else v for v in frag.score_molecule(q['spectra'], smiles, cache={})])
        ns = C.neural_scores(scorer, q, smiles)
        base = C.zscore(np.nan_to_num(fs, nan=-1.0)) + C.zscore(ns)
        idx = list(np.argsort(-base, kind='stable')[:short])
        base_rank = {keys[i]: r for r, i in enumerate(np.argsort(-base, kind='stable'), 1)}
        fs, ns = fs[idx], ns[idx]
    else:
        pos = {k: i for i, k in enumerate(keys)}
        idx = [pos[k] for k in order[:short]]
        base_rank = {keys[i]: r for r, i in enumerate(idx, 1)}
        sm = [smiles[i] for i in idx]
        fs = np.array([np.nan if v is None else v for v in frag.score_molecule(q['spectra'], sm, cache={})])
        ns = C.neural_scores(scorer, q, sm)
    fs = np.nan_to_num(fs, nan=-1.0)
    sk = [keys[i] for i in idx]
    mols = [Chem.MolFromSmiles(smiles[i]) for i in idx]
    fps = [C.FPG.GetFingerprint(m) if m is not None else None for m in mols]
    zf, zn = C.zscore(fs), C.zscore(ns)
    rf, rn = _rank(fs), _rank(ns)

    out = {}
    for thr, nbs in neighbours.items():
        nfps = [C.FPG.GetFingerprint(m) if (m := Chem.MolFromSmiles(s)) is not None else None for _, s, _ in nbs]
        sup = []
        for k, fp in zip(sk, fps):
            use = [(fp2, sim) for (ik, _, sim), fp2 in zip(nbs, nfps) if ik != k and fp2 is not None][:K_NEIGHBOURS]
            if fp is None or not use:
                sup.append((0.0, 0.0, 0.0, 0.0))
                continue
            tan = np.array(DataStructs.BulkTanimotoSimilarity(fp, [u[0] for u in use]))
            sim = np.array([u[1] for u in use])
            sup.append((float((tan * sim).max()), float(tan[0]), float(tan[:5].mean()),
                        float((tan * sim).sum() / max(sim.sum(), 1e-9))))
        sup = np.array(sup)
        zs = C.zscore(sup[:, 3])
        best_nb = nbs[0][2] if nbs else 0.0
        rows = []
        for j, k in enumerate(sk):
            rows.append(dict(
                cand=k, smiles=smiles[idx[j]], frag=fs[j], neural=ns[j], z_frag=zf[j], z_neural=zn[j],
                rank_frag=int(rf[j]), rank_neural=int(rn[j]), base_rank=j + 1, n_short=len(sk),
                heavy=mols[j].GetNumHeavyAtoms() if mols[j] is not None else 0, sup_max=sup[j, 0], sup_top1=sup[j, 1],
                sup_mean5=sup[j, 2], sup_wsum=sup[j, 3], z_sup=zs[j], best_nb_sim=best_nb))
        out[thr] = rows
    return out, base_rank
