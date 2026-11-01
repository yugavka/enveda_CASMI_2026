"""Untrained MetFrag-lite re-scorer: how much MS/MS intensity a candidate's <=2-bond-cleavage fragments explain.

Independent of pv.py (same idea as pv.fragment_masses / explain_score_adduct, new code): enumerate the neutral masses
of the connected pieces left after removing 1 or 2 bonds of the heavy-atom graph (implicit H stay on their atom),
remember the fewest cleavages that give each mass, then match every peak against
`neutral fragment + dh*H + charge carrier` for dh in `h_shifts` and the carriers the precursor adduct allows.
A peak's credit is the weight of its cheapest explanation (1 for 0/1 cleavages, `w2` for 2, `w3` for 3),
and a spectrum's score is the credited share of its (weighted) intensity. Defaults follow the forum ablation on a
241-structure same-formula isomer panel (+0.065 MRR over the common baseline): w2=0.6, linear intensity,
tol 0.005 Da, H shifts -2..+3, at most 2 cleavages. `BASELINE_CFG` is the common/pv.py-style setting.

casmi26 fork: optional ppm tolerance, ring-opening credit (`w_ring`, default from env CASMI_FRAG_W_RING),
spectrum aggregation and precursor removal; with all of them off it scores exactly like the upstream file.

    score_molecule(spectra, cand_smiles, cfg=None, cache=None) -> list[float | None]

`spectra`: list of dicts {mz: ndarray, it: ndarray (base peak = 1), adduct: str, prec: float, mode: +1/-1}.
Returns one score per candidate (mean over the spectra that have peaks; higher = better), None if the SMILES does
not parse. Fragment masses are cached per SMILES in `cache` (a dict; pass one in to reuse it across molecules).
The enumeration is numba-compiled when numba is importable (pure-Python fallback otherwise, ~100x slower).
"""
from __future__ import annotations

import os

import numpy as np


def _env_weight(name):
    """Optional float from the environment ('' / unset / 'none' -> None), so a notebook switch reaches subprocesses."""
    v = os.environ.get(name, '').strip().lower()
    return None if v in ('', 'none') else float(v)

# monoisotopic masses (same values as pv.AMU); other elements come from RDKit's periodic table
AMU = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196, 'P': 30.97376163,
       'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473,
       'Na': 22.9897692809, 'K': 38.96370668, 'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213}
H = AMU['H']
E = 0.00054857990
PROTON = H - E
H2O = 2 * H + AMU['O']
NH4 = AMU['N'] + 4 * H
FORMATE = AMU['C'] + 2 * H + 2 * AMU['O']

# charge carriers a fragment ion may keep, per precursor adduct (added to the neutral fragment mass)
CARRIERS = {
    '[M+H]+': (PROTON,), '[M-H2O+H]+': (PROTON,), '[M-2H2O+H]+': (PROTON,),
    '[M+NH4]+': (PROTON, NH4 - E), '[M+Na]+': (PROTON, AMU['Na'] - E), '[M+K]+': (PROTON, AMU['K'] - E),
    '[M-H]-': (-PROTON,), '[M-H2O-H]-': (-PROTON,), '[M+CH2O2-H]-': (-PROTON, FORMATE - PROTON),
    '[M+Cl]-': (-PROTON, AMU['Cl'] + E),
}

DEFAULT_CFG = dict(
    w1=1.0,                      # credit for a peak explained by the precursor or a 1-bond cleavage
    w2=0.6,                      # credit for a peak only explained by a 2-bond cleavage
    w3=0.3,                      # credit for 3-bond cleavages (only used when max_breaks=3)
    intensity='linear',          # 'linear' | 'sqrt' | a float power applied to `it`
    mz_power=0.0,                # extra peak weight (mz / prec) ** mz_power (0 = off)
    tol=0.005,                   # absolute m/z tolerance, Da
    h_shifts=tuple(range(-2, 4)),  # hydrogen shifts dh applied to neutral fragments (-2..+3)
    max_breaks=2,                # bond cleavages per fragment (1..3)
    max_bonds_2=100,             # above this many bonds only 1-bond cleavages are enumerated (cost cap)
    max_bonds_3=40,              # above this many bonds 3-bond cleavages are skipped
    # extensions (casmi26 fork); the defaults below reproduce the upstream scorer exactly
    ppm=0.0,                     # if > 0, tolerance per peak = max(tol, ppm * 1e-6 * mz)
    # credit for 2 cleavages inside one ring (one ring-opening event, e.g. RDA); None = w2.
    # Read from CASMI_FRAG_W_RING (bench: 0.8 -> +0.016 MRR on enveda-np-examples, -0.008 on enveda-180)
    w_ring=_env_weight('CASMI_FRAG_W_RING'),
    agg='mean',                  # 'mean' | 'npeaks' (spectra weighted by sqrt(peak count)) | 'max'
    drop_prec=0.0,               # if > 0, ignore peaks within this many Da below the precursor and above it
)
# the common baseline (pv.py-style): all cleavages equal, sqrt intensity, 0.01 Da, H shifts -2..+2
BASELINE_CFG = dict(DEFAULT_CFG, w2=1.0, intensity='sqrt', tol=0.01, h_shifts=tuple(range(-2, 3)), w_ring=None)


def resolve_cfg(cfg=None):
    """DEFAULT_CFG updated with `cfg` (unknown keys raise, so typos in variants do not pass silently)."""
    out = dict(DEFAULT_CFG)
    for k, v in (cfg or {}).items():
        if k not in out:
            raise KeyError(f'frag_rescore: unknown cfg key {k!r}')
        out[k] = v
    out['h_shifts'] = tuple(int(d) for d in out['h_shifts'])
    if not 1 <= int(out['max_breaks']) <= 3:
        raise ValueError('frag_rescore: max_breaks must be 1, 2 or 3')
    if out['agg'] not in ('mean', 'npeaks', 'max'):
        raise ValueError('frag_rescore: agg must be mean, npeaks or max')
    return out


# ---------------------------------------------------------------------------------------------------------------
# fragment enumeration (numba kernel with a pure-Python fallback: same source)
# ---------------------------------------------------------------------------------------------------------------
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, k):
    """Union the kept bonds; if the graph splits, append each piece's mass with cleavage count k."""
    for a in range(n):
        parent[a] = a
        acc[a] = 0.0
    ncomp = n
    for e in range(ea.shape[0]):
        if removed[e]:
            continue
        ra = _find(parent, ea[e])
        rb = _find(parent, eb[e])
        if ra != rb:
            parent[ra] = rb
            ncomp -= 1
    if ncomp < 2:
        return cnt
    for a in range(n):
        r = _find(parent, a)
        acc[r] += w[a]
    for a in range(n):
        if parent[a] == a:
            out_m[cnt] = acc[a]
            out_k[cnt] = k
            cnt += 1
    return cnt


def _enumerate(n, w, ea, eb, max_breaks, base_components, same_ring):
    """All piece masses after removing 1..max_breaks bonds (k = number removed); row 0.. are the intact pieces.

    A 2-bond cleavage whose bonds share a ring (`same_ring[i, j]`) is emitted with k = 4 (ring-opening event).
    """
    nb = ea.shape[0]
    n2 = nb * (nb - 1) // 2 if max_breaks >= 2 else 0
    n3 = nb * (nb - 1) * (nb - 2) // 6 if max_breaks >= 3 else 0
    cap = base_components + (base_components + 1) * nb + (base_components + 2) * n2 + (base_components + 3) * n3
    out_m = np.empty(cap, np.float64)
    out_k = np.empty(cap, np.int8)
    parent = np.empty(n, np.int64)
    acc = np.empty(n, np.float64)
    removed = np.zeros(nb, np.bool_)
    cnt = 0
    for i in range(nb):
        removed[i] = True
        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 1)
        if max_breaks >= 2:
            for j in range(i + 1, nb):
                removed[j] = True
                cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt,
                                       4 if same_ring[i, j] else 2)
                if max_breaks >= 3:
                    for l in range(j + 1, nb):
                        removed[l] = True
                        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 3)
                        removed[l] = False
                removed[j] = False
        removed[i] = False
    return out_m[:cnt], out_k[:cnt]


try:  # numba is on the Kaggle image and in requirements.txt; NUMBA_CACHE_DIR is set by nb/01_setup.py
    from numba import njit
    _find = njit(cache=True, nogil=True)(_find)
    _emit_components = njit(cache=True, nogil=True)(_emit_components)
    _enumerate = njit(cache=True, nogil=True)(_enumerate)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback path
    HAVE_NUMBA = False


def same_ring_matrix(smi, nb):
    """nb x nb bool: bonds i and j lie in one SSSR ring (all False if the SMILES does not parse)."""
    from rdkit import Chem
    out = np.zeros((nb, nb), np.bool_)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return out
    for ring in m.GetRingInfo().BondRings():
        ix = np.array(ring, np.int64)
        out[np.ix_(ix, ix)] = True
    np.fill_diagonal(out, False)
    return out


def mol_graph(smi):
    """(atom masses incl. implicit H, bond begin idx, bond end idx, n connected pieces) or None if unparsable."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    except Exception:
        m = None
    if m is None or m.GetNumAtoms() == 0:
        return None
    pt = Chem.GetPeriodicTable()
    w = np.empty(m.GetNumAtoms(), np.float64)
    for a in m.GetAtoms():
        s = a.GetSymbol()
        am = AMU.get(s)
        if am is None:
            am = pt.GetMostCommonIsotopeMass(s)
        w[a.GetIdx()] = am + a.GetTotalNumHs() * H
    ea = np.array([b.GetBeginAtomIdx() for b in m.GetBonds()], np.int64)
    eb = np.array([b.GetEndAtomIdx() for b in m.GetBonds()], np.int64)
    return w, ea, eb, len(Chem.GetMolFrags(m))


def fragment_levels(smi, max_breaks=2, max_bonds_2=100, max_bonds_3=40, ring=False):
    """Sorted neutral fragment masses grouped by the fewest cleavages that produce them: [k=0, k=1, ..., k=max].

    k=0 holds the intact molecule (sum of its pieces, and each piece if the SMILES has several). None if the
    SMILES does not parse. With `ring`, 2-bond cleavages inside one ring form an extra last level, ranked
    between k=1 and k=2 (a mass reachable both ways counts as the ring event).
    """
    g = mol_graph(smi)
    if g is None:
        return None
    w, ea, eb, npieces = g
    nb = len(ea)
    mb = int(max_breaks)
    if mb >= 3 and nb > max_bonds_3:
        mb = 2
    if mb >= 2 and nb > max_bonds_2:
        mb = 1
    levels = [np.array([w.sum()])]
    if npieces > 1:  # salts / mixtures: each intact piece also counts as uncleaved
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        levels = [np.unique(np.concatenate([[w.sum()], [w[list(ix)].sum() for ix in Chem.GetMolFrags(m)]]))]
    codes = list(range(int(max_breaks) + 1)) + ([4] if ring else [])
    if nb == 0:
        return levels + [np.zeros(0) for _ in codes[1:]]
    same_ring = same_ring_matrix(smi, nb) if ring and mb >= 2 else np.zeros((nb, nb), np.bool_)
    fm, fk = _enumerate(len(w), w, ea, eb, mb, npieces, same_ring)
    fm = np.concatenate([levels[0], fm])
    fk = np.concatenate([np.zeros(len(levels[0]), np.int8), fk])
    cost = np.array([0, 1, 3, 4, 2], np.int8)[fk]  # ring event (k=4) sorts between 1 and 2 cleavages
    key = np.round(fm, 5)
    order = np.lexsort((cost, key))                # by mass, then fewest cleavages first
    key, fm, fk = key[order], fm[order], fk[order]
    first = np.ones(len(key), bool)
    first[1:] = key[1:] != key[:-1]
    fm, fk = fm[first], fk[first]
    return [np.sort(fm[fk == k]) for k in codes]


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------
def _carriers(adduct, mode):
    c = CARRIERS.get(adduct)
    if c is not None:
        return c
    if isinstance(adduct, str) and adduct.endswith('-'):
        return (-PROTON,)
    if isinstance(adduct, str) and adduct.endswith('+'):
        return (PROTON,)
    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)


def _prep_spectrum(s, c):
    """Peak weights and the neutral masses each peak would correspond to, for every (carrier, dh) pair."""
    mz = np.asarray(s.get('mz', ()), np.float64).ravel()
    it = np.asarray(s.get('it', ()), np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if c['drop_prec'] and s.get('prec'):
        keep = mz < float(s['prec']) - float(c['drop_prec'])
        mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return None
    p = c['intensity']
    iw = it if p == 'linear' else np.sqrt(it) if p == 'sqrt' else it ** float(p)
    if c['mz_power']:
        prec = float(s.get('prec') or mz.max())
        iw = iw * (mz / prec) ** float(c['mz_power'])
    tot = iw.sum()
    if not tot > 0:
        return None
    shifts = np.array([car + dh * H for car in _carriers(s.get('adduct'), s.get('mode')) for dh in c['h_shifts']])
    q = (mz[None, :] - shifts[:, None]).ravel()  # neutral mass needed, per (shift, peak)
    tol = float(c['tol'])
    if c['ppm']:
        tol = np.tile(np.maximum(tol, float(c['ppm']) * 1e-6 * mz), len(shifts))
    return iw / tot, q, len(shifts), len(mz), tol


def _score_spectrum(levels, weights, prep):
    iw, q, ns, npk, tol = prep
    best = np.zeros(npk)
    for arr, wt in zip(levels, weights):
        if len(arr) == 0 or wt <= 0:
            continue
        a = np.searchsorted(arr, q - tol, 'left')
        b = np.searchsorted(arr, q + tol, 'right')
        hit = (b > a).reshape(ns, npk).any(axis=0)
        best = np.where(hit & (wt > best), wt, best)
    return float(iw @ best)


def candidate_levels(smi, cfg=None, cache=None):
    """Cached fragment_levels for one SMILES under `cfg` (only the enumeration keys matter)."""
    return _levels_cached(smi, resolve_cfg(cfg), cache)


def _levels_cached(smi, c, cache):
    ring = c['w_ring'] is not None
    key = (smi, int(c['max_breaks']), int(c['max_bonds_2']), int(c['max_bonds_3'])) + (('ring',) if ring else ())
    if cache is not None and key in cache:
        return cache[key]
    try:
        lv = fragment_levels(smi, c['max_breaks'], c['max_bonds_2'], c['max_bonds_3'], ring=ring)
    except Exception:
        lv = None
    if cache is not None:
        cache[key] = lv
    return lv


def score_molecule(spectra, cand_smiles, cfg=None, cache=None):
    """One score per candidate: mean explained-intensity share over the molecule's spectra (None: unparsable)."""
    c = resolve_cfg(cfg)
    cache = {} if cache is None else cache
    weights = [1.0, float(c['w1']), float(c['w2']), float(c['w3'])][: int(c['max_breaks']) + 1]
    if c['w_ring'] is not None:
        weights.append(float(c['w_ring']))
    preps = [p for p in (_prep_spectrum(s, c) for s in (spectra or [])) if p is not None]
    sw = np.sqrt([p[3] for p in preps]) if c['agg'] == 'npeaks' else None
    out = []
    for smi in cand_smiles:
        lv = _levels_cached(smi, c, cache)
        if lv is None:
            out.append(None)
            continue
        if not preps:
            out.append(0.0)
            continue
        per = [_score_spectrum(lv, weights, p) for p in preps]
        out.append(float(np.max(per) if c['agg'] == 'max' else np.average(per, weights=sw)))
    return out


def warmup():
    """Compile the numba kernel once (a few seconds) so the first real molecule is not charged for it."""
    fragment_levels('CCOC(=O)c1ccccc1O')
    return HAVE_NUMBA
