"""Spectrum-candidate scorer compatible with Saner Turhaner's casmi_spectrum_model_v1 checkpoints (Apache-2.0).

Our own implementation of the architecture and features described by that bundle, so that only data is read from
it: config.json and a weights-only checkpoint (torch.load(..., weights_only=True)). No bundle code is executed.

    Scorer(bundle_dir, checkpoint).score_candidates(spectra, smiles) -> np.ndarray (larger = better, -inf invalid)

spectra: dicts with mzs, intensities, precursor_mz, adduct, ionization_mode, collision_energy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ADDUCTS = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+K]+', '[M+NH4]+', '[M+Cl]-', '[M+FA-H]-', '[M+H-H2O]+', '[M+2H]2+',
           '[M-2H]2-']


class SpectrumCandidateModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['width']
        self.cfg = cfg
        self.register_buffer('freq', 2. ** torch.arange(10).float())
        self.peak = nn.Linear(44, d)
        self.meta = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.adduct = nn.Embedding(11, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, cfg['heads'], d * 4, cfg['dropout'], batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.spec_projection = nn.Linear(d, cfg['embedding'])
        self.fp_head = nn.Linear(d, cfg['fp_bits'])
        self.molecule = nn.Sequential(nn.Linear(cfg['fp_bits'], 512), nn.GELU(), nn.Dropout(cfg['dropout']),
                                      nn.Linear(512, cfg['embedding']))

    def encode_spectra(self, x, lengths, meta, adduct):
        a = x[:, :, [0, 3]].unsqueeze(-1) * self.freq * torch.pi
        p = torch.cat([x, a.sin().flatten(2), a.cos().flatten(2)], -1)
        tokens = self.peak(p)
        cls = self.cls.expand(len(x), -1, -1) + self.meta(meta)[:, None] + self.adduct(adduct)[:, None]
        tokens = torch.cat([cls, tokens], 1)
        mask = torch.arange(x.shape[1] + 1, device=x.device)[None] > lengths[:, None]
        h = self.norm(self.transformer(tokens, src_key_padding_mask=mask)[:, 0])
        return F.normalize(self.spec_projection(h), dim=-1), self.fp_head(h)

    def encode_molecules(self, fp):
        return F.normalize(self.molecule(fp), dim=-1)


def peaks(mz, it, precursor, max_peaks=96):
    mz, it = np.asarray(mz, np.float32).ravel(), np.asarray(it, np.float32).ravel()
    keep = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if not len(mz):
        raise ValueError('No valid peaks')
    ix = np.argsort(-it, kind='stable')[:max_peaks]
    ix = ix[np.argsort(mz[ix], kind='stable')]
    mz, it = mz[ix], it[ix]
    it = it / it.max()
    out = np.zeros((max_peaks, 4), np.float32)
    out[:len(mz)] = np.stack([mz / 1000, np.sqrt(it), np.log1p(100 * it) / np.log(101), (float(precursor) - mz) / 1000], 1)
    return out, len(mz)


def metadata(precursor, adduct, mode, ce):
    try:
        a = np.asarray(ce if ce is not None else [], np.float32).ravel()
        a = a[np.isfinite(a)]
        ev, known = (float(a.mean()), 1.) if len(a) else (0., 0.)
    except (ValueError, TypeError):
        ev, known = 0., 0.
    pol = 1. if str(mode).lower() == 'positive' else -1.
    return (np.array([float(precursor) / 1000, np.clip(ev, 0, 500) / 100, known, pol], np.float32),
            ADDUCTS.index(adduct) + 1 if adduct in ADDUCTS else 0)


def fingerprint(smiles, nbits):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits).GetFingerprintAsNumPy(mol)


class Scorer:
    def __init__(self, bundle, checkpoint='final_model.pt'):
        root = Path(bundle)
        self.cfg = json.loads((root / 'config.json').read_text())
        self.model = SpectrumCandidateModel(self.cfg)
        obj = torch.load(root / checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(obj['state_dict'])
        self.model.eval()
        self.cache = {}

    @torch.inference_mode()
    def score_candidates(self, spectra, candidates):
        rec = []
        for s in spectra:
            p, n = peaks(s['mzs'], s['intensities'], s['precursor_mz'], self.cfg['max_peaks'])
            m, a = metadata(s['precursor_mz'], s.get('adduct', ''), s.get('ionization_mode', ''), s.get('collision_energy'))
            rec.append((p, n, m, a))
        t = [torch.as_tensor(np.array([r[j] for r in rec])) for j in range(4)]
        z, p = self.model.encode_spectra(t[0].float(), t[1].long(), t[2].float(), t[3].long())
        z, p = z.mean(0), p.mean(0)  # equal to the averaged per-spectrum score
        out = np.full(len(candidates), -np.inf, np.float32)
        valid, fps = [], []
        for i, s in enumerate(candidates):
            if s not in self.cache:
                self.cache[s] = fingerprint(s, self.cfg['fp_bits'])
            if self.cache[s] is not None:
                valid.append(i)
                fps.append(self.cache[s])
        if valid:
            fp = torch.as_tensor(np.array(fps), dtype=torch.float32)
            score = (self.model.encode_molecules(fp) @ z) / self.cfg['temperature'] \
                + self.cfg['fp_score_weight'] * (fp @ p) / self.cfg['fp_bits']
            out[valid] = score.numpy()
        return out
