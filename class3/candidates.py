"""Shared class-3 candidate machinery for the bench and the Kaggle notebook stage.

neighbour_scores: spectral similarity of a query to every library structure.
build: generate candidates from the query's neighbours, score them (fragment scorer, neural scorer) and compute the
ranker features. The bench and the notebook call exactly this code, so their features match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import library as L
import transforms as T

RDLogger.DisableLog('rdApp.*')
CAP_PER_NEIGHBOUR = 150
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# our adduct names -> the neural model's vocabulary (sanpier casmi_spectrum_model_v1 features.ADDUCTS)
NEURAL_ADDUCT = {'[M+CH2O2-H]-': '[M+FA-H]-', '[M-H2O+H]+': '[M+H-H2O]+'}


def k14(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def zscore(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return np.nan_to_num((x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x), nan=-3.0)


def neighbour_scores(q, lib_meta, frag, nl):
    """Best combined cosine of each library structure to any of the query's spectra (same polarity only)."""
    best = np.zeros(len(lib_meta))
    pos = (lib_meta.ionization_mode == 'positive').to_numpy()
    for s in q['spectra']:
        fb, fw, nb, nw = L.spectrum_vectors(s['mz'], s['it'], s['prec'], s['adduct'])
        if len(fb) < 3:
            continue
        qf = np.zeros(L.NBINS); qf[fb] = fw
        qn = np.zeros(L.NBINS); qn[nb] = nw
        sim = 0.5 * (frag @ qf + nl @ qn)
        sim[pos != (s['mode'] > 0)] = 0.0
        best = np.maximum(best, sim)
    return pd.Series(best).groupby(lib_meta.inchikey14.to_numpy()).max()


def neural_scores(scorer, q, smiles):
    if scorer is None:
        return np.full(len(smiles), np.nan)
    spectra = [dict(mzs=s['mz'], intensities=s['it'], precursor_mz=s['prec'],
                    adduct=NEURAL_ADDUCT.get(s['adduct'], s['adduct']),
                    ionization_mode='positive' if s['mode'] > 0 else 'negative', collision_energy=s.get('ce'))
               for s in q['spectra']]
    out = scorer.score_candidates(spectra, smiles).astype(float)
    out[~np.isfinite(out)] = np.nan
    return out


def build(q, neighbours, steps, scorer, qid='', truth=None):
    """Candidates generated from `neighbours` [(smiles, similarity), ...] best first.

    Returns (keys, raw, feats): candidate InChIKey14 blocks; raw per-candidate arrays (frag, neural, parent_sim,
    parent_rank, z_frag, z_neural, z_parent); one feature dict per candidate (label = key == truth).
    """
    import frag_rescore as frag
    cands = {}    # key -> (smiles, parent rank, parent similarity, rule names, parent smiles)
    support = {}  # key -> number of neighbours that generate it
    for rank, (smi, sim) in enumerate(neighbours, 1):
        for c, plan in T.generate_plans(smi, q['mass'], max_steps=steps, cap=CAP_PER_NEIGHBOUR).items():
            key = k14(c)
            if not key:
                continue
            support[key] = support.get(key, 0) + 1
            if key not in cands:
                cands[key] = (c, rank, sim, plan, smi)
    if not cands:
        return [], {}, []
    keys = list(cands)
    smiles = [cands[k][0] for k in keys]
    fs = np.array([np.nan if v is None else v for v in frag.score_molecule(q['spectra'], smiles, cache={})])
    fs = np.nan_to_num(fs, nan=-1.0)
    psim = np.array([cands[k][2] for k in keys])
    prank = np.array([cands[k][1] for k in keys])
    ns = neural_scores(scorer, q, smiles)
    zf, zn, zp = zscore(fs), zscore(ns), zscore(psim)
    raw = dict(frag=fs, neural=ns, parent_sim=psim, parent_rank=prank, z_frag=zf, z_neural=zn, z_parent=zp)

    rank_of = lambda v: (-np.nan_to_num(v, nan=-1e9)).argsort().argsort() + 1
    rf, rn = rank_of(fs), rank_of(ns)
    feats = []
    for j, k in enumerate(keys):
        smi, _, _, plan, parent = cands[k]
        m, pm = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(parent)
        tan = DataStructs.TanimotoSimilarity(FPG.GetFingerprint(m), FPG.GetFingerprint(pm)) if m and pm else np.nan
        feats.append(dict(
            qid=qid, cand=k, label=int(k == truth), smiles=smi,
            frag=fs[j], neural=ns[j], parent_sim=psim[j], parent_rank=int(prank[j]), z_frag=zf[j], z_neural=zn[j],
            z_parent=zp[j], rank_frag=int(rf[j]), rank_neural=int(rn[j]), support=support[k], n_steps=len(plan),
            rule1=plan[0], rule2=plan[1] if len(plan) > 1 else '', tan_parent=tan,
            heavy=m.GetNumHeavyAtoms() if m else 0, heavy_parent=pm.GetNumHeavyAtoms() if pm else 0,
            n_cands=len(keys)))
    return keys, raw, feats
