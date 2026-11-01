"""Same-formula isomer bench for frag_rescore variants.

For each panel (one or more source libraries) we sample structures that have spectra with a test adduct, take up
to MAX_SPECTRA of their spectra as one query molecule, and rank the true structure among up to N_DECOYS training
structures with the same molecular formula, by fragment score alone. Reports MRR / top-1 per variant and a paired
bootstrap (over structures) of the MRR difference against the reference variant.

    .venv/bin/python -I frag/bench.py --data data/train.parquet --out frag/results
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frag_rescore as frag  # noqa: E402

TEST_ADDUCTS = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+',
                '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-']
PANELS = {
    'np_examples': ['enveda-np-examples'],      # timsTOF natural products; the upstream defaults were tuned here
    'gnps': ['gnps'],
    'riken': ['riken'],
    'mona_msdial': ['mona', 'msdial'],
    'massbank': ['massbank'],
    'enveda180': ['enveda-180'],                # timsTOF, synthetic chemistry
}
MAX_SPECTRA = 4
N_DECOYS = 30
TOP_PEAKS = 120

# variants: name -> cfg overrides on top of frag.DEFAULT_CFG (the upstream defaults); 'default' is the reference
VARIANTS = {
    'default': {},
    'baseline_pv': dict(frag.BASELINE_CFG),
    'tol003': dict(tol=0.003),
    'ppm10': dict(tol=0.002, ppm=10.0),
    'ppm15': dict(tol=0.003, ppm=15.0),
    'ppm20': dict(tol=0.003, ppm=20.0),
    'ring07': dict(w_ring=0.7),
    'ring08': dict(w_ring=0.8),
    'ring09': dict(w_ring=0.9),
    'ring10': dict(w_ring=1.0),
    'ring08_ppm10': dict(w_ring=0.8, tol=0.002, ppm=10.0),
    'agg_npeaks': dict(agg='npeaks'),
    'agg_max': dict(agg='max'),
    'drop_prec': dict(drop_prec=0.5),
}


def load_pool(path):
    """inchikey14 -> (smiles, formula) for every training structure, and formula -> [inchikey14]."""
    t = ds.dataset(path).to_table(columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).to_pandas()
    t = t.drop_duplicates('inchikey14')
    by_formula = t.groupby('molecular_formula').inchikey14.apply(list).to_dict()
    smiles = dict(zip(t.inchikey14, t.normalized_smiles))
    formula = dict(zip(t.inchikey14, t.molecular_formula))
    return smiles, formula, by_formula


def build_panel(path, libs, n_struct, rng, formula, by_formula):
    """Sample query molecules for one panel: list of dicts {ik, smiles, spectra, decoys}."""
    dset = ds.dataset(path)
    flt = ds.field('ingest_lib').isin(libs) & ds.field('adduct').isin(TEST_ADDUCTS)
    light = dset.to_table(columns=['inchikey14'], filter=flt).to_pandas()
    iks = [ik for ik in light.inchikey14.unique() if len(by_formula.get(formula.get(ik), ())) >= 6]
    iks = list(rng.choice(sorted(iks), size=min(n_struct, len(iks)), replace=False))
    cols = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ionization_mode', 'ms2_mzs',
            'ms2_normalized_intensities']
    heavy = dset.to_table(columns=cols, filter=flt & ds.field('inchikey14').isin(iks)).to_pandas()
    out = []
    for ik, g in heavy.groupby('inchikey14', sort=True):
        g = g[g.ms2_mzs.map(len) >= 5]
        if g.empty:
            continue
        g = g.sample(n=min(MAX_SPECTRA, len(g)), random_state=int(rng.integers(1 << 31)))
        spectra = []
        for r in g.itertuples(index=False):
            mz = np.asarray(r.ms2_mzs, float)
            it = np.asarray(r.ms2_normalized_intensities, float)
            keep = np.argsort(it)[-TOP_PEAKS:]
            spectra.append(dict(mz=mz[keep], it=it[keep], adduct=r.adduct, prec=float(r.precursor_mz),
                                mode=1 if str(r.ionization_mode).lower() == 'positive' else -1))
        others = [d for d in by_formula[formula[ik]] if d != ik]
        decoys = list(rng.choice(sorted(others), size=min(N_DECOYS, len(others)), replace=False))
        out.append(dict(ik=ik, smiles=g.normalized_smiles.iloc[0], spectra=spectra, decoys=decoys))
    return out


_SMILES = None


def _init(smiles):
    global _SMILES
    _SMILES = smiles
    frag.warmup()


def _score_query(args):
    """Reciprocal rank of the truth (ties count half) and top-1 flag, per variant."""
    q, variants = args
    cands = [q['smiles']] + [_SMILES[d] for d in q['decoys']]
    cache = {}
    res = {}
    for name, cfg in variants.items():
        s = np.array([np.nan if v is None else v for v in frag.score_molecule(q['spectra'], cands, cfg=cfg, cache=cache)])
        if not np.isfinite(s[0]):
            return q['ik'], None
        d = s[1:][np.isfinite(s[1:])]
        rank = 1 + (d > s[0]).sum() + 0.5 * (d == s[0]).sum()
        res[name] = (1.0 / rank, float(rank == 1))
    return q['ik'], res


def bootstrap_delta(a, b, rng, n=2000):
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train.parquet')
    ap.add_argument('--out', default='frag/results')
    ap.add_argument('--n', type=int, default=250, help='structures per panel')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--panels', default=','.join(PANELS))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = {k: VARIANTS[k] for k in a.variants.split(',')}
    t0 = time.time()
    smiles, formula, by_formula = load_pool(a.data)
    print(f'pool {len(smiles):,} structures, {len(by_formula):,} formulas ({time.time() - t0:.0f}s)', flush=True)
    rows, summary = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(smiles,)) as pool:
        for panel in a.panels.split(','):
            rng = np.random.default_rng([a.seed, sum(map(ord, panel))])
            qs = build_panel(a.data, PANELS[panel], a.n, rng, formula, by_formula)
            t1 = time.time()
            for ik, res in pool.map(_score_query, [(q, variants) for q in qs], chunksize=4):
                if res is None:
                    continue
                for name, (rr, top1) in res.items():
                    rows.append(dict(panel=panel, ik=ik, variant=name, rr=rr, top1=top1))
            df = pd.DataFrame([r for r in rows if r['panel'] == panel])
            piv = df.pivot(index='ik', columns='variant', values='rr')
            brng = np.random.default_rng(a.seed)
            for name in variants:
                mean, lo, hi = bootstrap_delta(piv[name], piv['default'], brng)
                summary.append(dict(panel=panel, variant=name, n=len(piv), mrr=piv[name].mean(),
                                    top1=df[df.variant == name].top1.mean(), d_mrr=mean, ci_lo=lo, ci_hi=hi))
            print(f'{panel}: {len(piv)} molecules, {time.time() - t1:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out / 'per_molecule.csv', index=False)
    s = pd.DataFrame(summary)
    s.to_csv(out / 'summary.csv', index=False)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), variants=variants, max_spectra=MAX_SPECTRA,
                                                     n_decoys=N_DECOYS, top_peaks=TOP_PEAKS), indent=2, default=str))
    pd.set_option('display.width', 200)
    print(s.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
