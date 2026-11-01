"""Compact structure-level spectral library for analog (neighbour) search.

Up to MAX_PER spectra per (inchikey14, ionisation mode) with a test adduct, from every source library except the
synthetic enveda-180. Each spectrum becomes two sparse unit vectors over 0.05 Da bins: fragment m/z and neutral loss
(neutral precursor mass - fragment m/z). Cached to data/cache/class3_library.npz + .parquet.

    .venv/bin/python -I class3/library.py --data data/train.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

ADDUCTS = {  # adduct -> (mass added to M per charge, multiplier of M); charge is always 1
    '[M+H]+': (1.007276, 1), '[M+NH4]+': (18.033823, 1), '[M-H2O+H]+': (-17.003289, 1),
    '[M-2H2O+H]+': (-35.013853, 1), '[M+Na]+': (22.989218, 1), '[M+K]+': (38.963158, 1),
    '[M-H]-': (-1.007276, 1), '[M-H2O-H]-': (-19.017841, 1), '[M+CH2O2-H]-': (44.998201, 1),
    '[M+Cl]-': (34.969402, 1),
}
BIN = 0.05
MAX_MZ = 2000.0
NBINS = int(MAX_MZ / BIN) + 1
TOP_PEAKS = 60
MAX_PER = 3
EXCLUDE_LIBS = ('enveda-180',)


def neutral_mass(prec, adduct):
    add, mult = ADDUCTS[adduct]
    return (prec - add) / mult


def spectrum_vectors(mz, it, prec, adduct):
    """(frag_bins, frag_w, nl_bins, nl_w) for one spectrum: top peaks, sqrt intensity, unit norm, summed per bin."""
    mz = np.asarray(mz, float)
    it = np.asarray(it, float)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz < prec - 0.5)  # drop the precursor region
    mz, it = mz[ok], it[ok]
    if len(mz) > TOP_PEAKS:
        keep = np.argsort(it)[-TOP_PEAKS:]
        mz, it = mz[keep], it[keep]
    w = np.sqrt(it)
    m = neutral_mass(prec, adduct)
    out = []
    for vals in (mz, m - mz):
        b = np.round(vals / BIN).astype(np.int64)
        good = (b > 0) & (b < NBINS)
        bb, ww = np.unique(b[good], return_inverse=True)
        acc = np.zeros(len(bb))
        np.add.at(acc, ww, w[good])
        n = np.linalg.norm(acc)
        out += [bb, acc / n if n > 0 else acc]
    return out


def to_csr(rows):
    indptr = np.cumsum([0] + [len(b) for b, _ in rows])
    idx = np.concatenate([b for b, _ in rows]) if rows else np.zeros(0, np.int64)
    val = np.concatenate([w for _, w in rows]) if rows else np.zeros(0)
    return sp.csr_matrix((val, idx, indptr), shape=(len(rows), NBINS))


def build(data, out_dir):
    t0 = time.time()
    pf = pq.ParquetFile(data)
    light_cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'ionization_mode', 'adduct', 'num_peaks']
    light = pf.read(columns=light_cols).to_pandas()
    light['row'] = np.arange(len(light))
    sel = light[~light.ingest_lib.isin(EXCLUDE_LIBS) & light.adduct.isin(list(ADDUCTS)) & (light.num_peaks >= 5)]
    # prefer the commonest adducts and the richest spectra, MAX_PER per structure and mode
    pref = {'[M+H]+': 0, '[M-H]-': 0}
    sel = sel.assign(p=sel.adduct.map(lambda a: pref.get(a, 1)))
    sel = sel.sort_values(['inchikey14', 'ionization_mode', 'p', 'num_peaks'], ascending=[True, True, True, False])
    sel = sel.groupby(['inchikey14', 'ionization_mode'], sort=False).head(MAX_PER)
    wanted = set(sel.row.tolist())
    print(f'selected {len(sel):,} spectra of {sel.inchikey14.nunique():,} structures ({time.time() - t0:.0f}s)', flush=True)

    meta, frag, nl = [], [], []
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        idx = [r - start for r in range(start, start + n) if r in wanted]
        if idx:
            tab = pf.read_row_group(g, columns=['precursor_mz', 'adduct', 'ms2_mzs', 'ms2_normalized_intensities'])
            tab = tab.take(idx).to_pandas()
            for r, row in zip(idx, tab.itertuples(index=False)):
                fb, fw, nb, nw = spectrum_vectors(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, row.adduct)
                if len(fb) < 3:
                    continue
                frag.append((fb, fw))
                nl.append((nb, nw))
                meta.append(start + r)
        start += n
        print(f'row group {g + 1}/{pf.num_row_groups}: {len(meta):,} spectra ({time.time() - t0:.0f}s)', flush=True)

    m = light.set_index('row').loc[meta, ['inchikey14', 'normalized_smiles', 'ionization_mode', 'adduct', 'ingest_lib']]
    m = m.reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out_dir / 'class3_library.parquet')
    sp.save_npz(out_dir / 'class3_frag.npz', to_csr(frag))
    sp.save_npz(out_dir / 'class3_nl.npz', to_csr(nl))
    print(f'library: {len(m):,} spectra, {m.inchikey14.nunique():,} structures ({time.time() - t0:.0f}s)')


def load(out_dir):
    m = pd.read_parquet(out_dir / 'class3_library.parquet')
    return m, sp.load_npz(out_dir / 'class3_frag.npz'), sp.load_npz(out_dir / 'class3_nl.npz')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train.parquet')
    ap.add_argument('--out', default='data/cache')
    a = ap.parse_args()
    build(a.data, Path(a.out))
