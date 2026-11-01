"""Mass-delta-driven biotransformations: turn a known neighbour structure into candidates of the query's mass.

Each rule is a reaction SMARTS with its monoisotopic mass change. For a neighbour of neutral mass m and a query of
neutral mass M, the rules (and pairs of rules) whose summed delta matches M - m within tolerance are applied at every
site; products are sanitised and de-duplicated.

    generate(neighbour_smiles, query_mass, ppm=10, abs_tol=0.003, max_steps=2) -> list[str]
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog('rdApp.*')

HEX = 'C1OC(CO)C(O)C(O)C1O'          # hexose (as an O-substituent)
PEN = 'C1OCC(O)C(O)C1O'              # pentose
DHX = 'C1OC(C)C(O)C(O)C1O'           # 6-deoxyhexose (rhamnose-like)
GLA = 'C1OC(C(=O)O)C(O)C(O)C1O'      # glucuronic acid

# (name, reaction SMARTS); the delta is computed from the SMARTS products below
RULES = [
    ('hydroxylation_ar', '[cH1:1]>>[c:1]O'),
    ('hydroxylation_al', '[CX4;H1,H2;!$(C[OX2]);!$(C[NX3]):1]>>[C:1]O'),
    ('dehydroxylation', '[CX4:1][OH1]>>[C:1]'),
    ('o_methylation', '[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]C'),
    ('n_methylation', '[NX3;H1,H2;!$(NC=O);!$(N-[a]):1]>>[N:1]C'),
    ('o_demethylation', '[OX2:1][CH3]>>[O:1]'),
    ('n_demethylation', '[NX3:1][CH3]>>[N:1]'),
    ('methoxylation_ar', '[cH1:1]>>[c:1]OC'),
    ('hydrogenation', '[C:1]=[C:2]>>[C:1][C:2]'),
    ('dehydrogenation', '[CX4;H1,H2;!R:1]-[CX4;H1,H2;!R:2]>>[C:1]=[C:2]'),
    ('dehydrogenation_ring', '[CX4;H1,H2;R:1]-[CX4;H1,H2;R:2]>>[C:1]=[C:2]'),
    ('ketone_from_alcohol', '[CX4;H1:1][OH1:2]>>[C:1]=[O:2]'),
    ('alcohol_from_ketone', '[CX3;!$(C[O,N;!R]);$(C([#6])[#6]):1]=[O:2]>>[C:1][O:2]'),
    ('hexosylation', f'[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]{HEX}'),
    ('pentosylation', f'[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]{PEN}'),
    ('deoxyhexosylation', f'[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]{DHX}'),
    ('glucuronidation', f'[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]{GLA}'),
    ('dehexosylation', '[#6:2][OX2:1][CH1]1O[CH1]([CH2][OH1])[CH1]([OH1])[CH1]([OH1])[CH1]1[OH1]>>[#6:2][O:1]'),
    ('depentosylation', '[#6:2][OX2:1][CH1]1O[CH2][CH1]([OH1])[CH1]([OH1])[CH1]1[OH1]>>[#6:2][O:1]'),
    ('dedeoxyhexosylation', '[#6:2][OX2:1][CH1]1O[CH1]([CH3])[CH1]([OH1])[CH1]([OH1])[CH1]1[OH1]>>[#6:2][O:1]'),
    ('acetylation', '[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]C(C)=O'),
    ('deacetylation', '[OX2:1]C(=O)[CH3]>>[O:1]'),
    ('malonylation', '[OX2H1;$(O[CX4]):1]>>[O:1]C(=O)CC(=O)O'),
    ('sulfation', '[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]S(=O)(=O)O'),
    ('prenylation_ar', '[cH1:1]>>[c:1]CC=C(C)C'),
    ('decarboxylation', '[#6:1]C(=O)[OH1]>>[#6:1]'),
    ('carboxylic_from_methyl', '[CH3;$(C[#6]):1]>>[C:1](=O)O'),
    ('hydration', '[C:1]=[C:2]>>[C:1][C:2]O'),
    ('dehydration', '[CX4;H1,H2:1][CX4:2][OH1]>>[C:1]=[C:2]'),
    # carbon skeleton: homologues (the commonest unmatched neighbour deltas on held-out GNPS: +-14, 28, 42, 56)
    ('c_methylation_al', '[CX4;H2,H3;!$(C[O,N,S]):1]>>[C:1]C'),
    ('c_methylation_ar', '[cH1:1]>>[c:1]C'),
    ('c_demethylation', '[#6:1][CH3]>>[#6:1]'),
    ('ethylation', '[CH3;$(C[CX4]):1]>>[C:1]CC'),
    ('propylation', '[CH3;$(C[CX4]):1]>>[C:1]CCC'),
    # heteroatom swaps and small acyl groups
    ('amine_to_hydroxyl', '[CX4,c:2][NH2]>>[*:2]O'),
    ('hydroxyl_to_amine', '[CX4,c:2][OH1]>>[*:2]N'),
    ('amide_to_acid', '[C:2](=O)[NH2]>>[C:2](=O)O'),
    ('acid_to_amide', '[C:2](=O)[OH1]>>[C:2](=O)N'),
    ('formylation', '[NX3;H1,H2;!$(NC=O):1]>>[N:1]C=O'),
    ('carbamoylation', '[OX2H1;$(O[#6]);!$(OC=O):1]>>[O:1]C(N)=O'),
    ('aza_ar', '[cH1;r6:1]>>[n:1]'),
]


def _delta(rule):
    """Mass change of a rule, measured on a probe molecule it applies to."""
    probes = ['OCC(O)Cc1ccccc1C', 'COc1ccc(CC=CCC(=O)C)cc1O', 'CN(C)CCc1ccc(OC(C)=O)cc1',
              'OC1C(O)C(O)C(CO)OC1Oc1ccccc1', 'OC1C(O)C(O)COC1Oc1ccccc1', 'OC1C(O)C(O)C(C)OC1Oc1ccccc1',
              'OC(=O)CCc1ccccc1', 'CNCCc1ccccc1', 'OCCC1CCCCC1C(C)=O', 'NCCc1ccccc1', 'NC(=O)CCc1ccccc1']
    rxn = AllChem.ReactionFromSmarts(rule)
    for p in probes:
        m = Chem.MolFromSmiles(p)
        for prods in rxn.RunReactants((m,)):
            q = prods[0]
            try:
                Chem.SanitizeMol(q)
            except Exception:
                continue
            return Descriptors.ExactMolWt(q) - Descriptors.ExactMolWt(m)
    raise ValueError(f'no probe for {rule}')


@lru_cache(maxsize=1)
def compiled():
    return [(name, AllChem.ReactionFromSmarts(s), _delta(s)) for name, s in RULES]


def plans(delta, tol, max_steps=2):
    """Rule-index tuples whose summed mass change matches `delta` within `tol`."""
    rules = compiled()
    out = []
    for k in range(1, max_steps + 1):
        for combo in combinations_with_replacement(range(len(rules)), k):
            if abs(sum(rules[i][2] for i in combo) - delta) <= tol:
                out.append(combo)
    return out


def _apply(rxn, smiles_list, cap):
    """Products of `rxn` on each input, in a deterministic order (the cap must not depend on hash seeds)."""
    out = {}
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for prods in rxn.RunReactants((m,)):
            q = prods[0]
            try:
                Chem.SanitizeMol(q)
                s = Chem.MolToSmiles(q)
            except Exception:
                continue
            if '.' not in s:
                out[s] = None
            if len(out) >= cap:
                return list(out)
    return list(out)


def generate_plans(neighbour, query_mass, ppm=10.0, abs_tol=0.003, max_steps=2, cap=400):
    """{candidate smiles: rule-name tuple of the first plan that produced it} for 1..max_steps rules."""
    m = Chem.MolFromSmiles(neighbour)
    if m is None:
        return {}
    base = Descriptors.ExactMolWt(m)
    tol = max(abs_tol, ppm * 1e-6 * query_mass)
    rules = compiled()
    root = Chem.MolToSmiles(m)
    out = {}
    for combo in plans(query_mass - base, tol, max_steps):
        cur = [root]
        for i in combo:
            cur = _apply(rules[i][1], cur, cap)
            if not cur:
                break
        for s in cur:
            out.setdefault(s, tuple(rules[i][0] for i in combo))
        if len(out) >= cap:
            break
    out.pop(root, None)
    return out


def generate(neighbour, query_mass, ppm=10.0, abs_tol=0.003, max_steps=2, cap=400):
    """Candidates of mass `query_mass` reachable from `neighbour` by 1..max_steps rules (neighbour itself excluded)."""
    return sorted(generate_plans(neighbour, query_mass, ppm, abs_tol, max_steps, cap))
