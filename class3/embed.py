"""Embed the class-3 insertion stage (class3/*.py + the trained ranker) into the Kaggle notebook.

Adds (or replaces) one code cell right after the final merge layer and the CLASS3 switches in the config cell.
Idempotent; re-run after changing any class3 module or retraining the ranker.

    .venv/bin/python -I class3/embed.py notebooks/casmi26-baseline-0477/casmi26-baseline-0477.ipynb
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ['library.py', 'transforms.py', 'neural.py', 'candidates.py', 'stage.py']
ASSETS = ['ranker/ranker.txt', 'ranker/meta.json']
EXTRA = {'features.py': '../rerank/features.py',  # isomer reranker (rerank/): published name -> source
         'rerank/ranker.txt': '../rerank/model/ranker.txt', 'rerank/meta.json': '../rerank/model/meta.json'}
MARKER = '# ── Class-3 analog insertion'
CONFIG_ANCHOR = 'OUTSIDER_FILL = True\n'
CONFIG_ADD = """
# Class-3 analog insertion (cell after the final layer; class3/ in our repo): for molecules whose best library match is
# below CLASS3_CFG['gate'], generate analogs of the spectral neighbours (41 mass-delta biotransformation rules), rank
# them with a LambdaRank model trained on held-out library molecules and insert the best n_insert at rank `position`.
CLASS3 = True
CLASS3_CFG = dict(gate=0.87, position=3, n_insert=3, seconds=2400)
"""

CELL = '''{marker} (ours): new structures for molecules the pipeline cannot confirm ───────────────────
# Sources: class3/ in our repository (library, transforms, neural, candidates, stage) and the LambdaRank ranker trained
# by class3/ranker.py. The neural scorer is our re-implementation of Saner Turhaner's casmi_spectrum_model_v1
# (Apache-2.0); only its config and weights are read from the attached dataset. Any error restores the input file.
import sys, json, time, shutil, traceback, glob
from pathlib import Path
C3_FILES = {files}
C3_SHA256 = {shas}
C3_DIR = CODE / 'class3'
for _name, _text in C3_FILES.items():
    (C3_DIR / _name).parent.mkdir(parents=True, exist_ok=True)
    assert __import__('hashlib').sha256(_text.encode()).hexdigest() == C3_SHA256[_name], _name
    (C3_DIR / _name).write_text(_text)
for _p in (str(C3_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASS3_REPORT = dict(status='disabled')
if globals().get('CLASS3', False):
    _sub = ROOT / 'submission.csv'
    _backup = ROOT / 'submission_before_class3.csv'
    shutil.copyfile(_sub, _backup)
    try:
        if (time.time() - NB_T0) / 3600 > TIME_GUARD_HOURS:
            CLASS3_REPORT = dict(status='skipped_time_guard')
        else:
            _bundle = sorted(glob.glob(str(INPUT / '**/casmi_spectrum_model_v1/config.json'), recursive=True))
            assert _bundle, 'casmi_spectrum_model_v1 bundle not attached'
            import stage as _c3
            CLASS3_REPORT = _c3.run(ROOT, COMP, Path(_bundle[0]).parent, globals().get('LIB_MAX') or {{}},
                                    CLASS3_CFG, ranker_dir=C3_DIR / 'ranker')
    except Exception as exc:
        traceback.print_exc()
        shutil.copyfile(_backup, _sub)
        CLASS3_REPORT = dict(status='restored_after_error', exception=repr(exc)[:600])
(ROOT / 'class3_report.json').write_text(json.dumps(CLASS3_REPORT, indent=2, default=str))
_final = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert _final.molecule_id.tolist() == SAMPLE_IDS
print('CLASS3', json.dumps({{k: v for k, v in CLASS3_REPORT.items() if k != 'cfg'}}, default=str))
'''


def src(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def main(path):
    path = Path(path)
    nb = json.loads(path.read_text())
    files = {name: (HERE / name).read_text() for name in MODULES + ASSETS}
    files.update({name: (HERE / src_path).read_text() for name, src_path in EXTRA.items()})
    shas = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()}
    cell_src = CELL.format(marker=MARKER, files=repr(files), shas=repr(shas))

    # config switches
    cfg = nb['cells'][1]
    c1 = src(cfg)
    if 'CLASS3 =' not in c1:
        assert c1.count(CONFIG_ANCHOR) == 1, 'config anchor missing'
        c1 = c1.replace(CONFIG_ANCHOR, CONFIG_ANCHOR + CONFIG_ADD)
        cfg['source'] = c1 if isinstance(cfg['source'], str) else c1.splitlines(keepends=True)

    # the stage cell: replace ours if present, else insert right after the final merge layer
    ours = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and src(c).startswith(MARKER)]
    new_cell = dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=cell_src)
    if ours:
        nb['cells'][ours[0]]['source'] = cell_src
        where = ours[0]
    else:
        merge = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'def pxm_merge' in src(c)]
        assert len(merge) == 1, merge
        where = merge[0] + 1
        nb['cells'].insert(where, new_cell)
    path.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')))
    print(f'class3 stage in cell {where}; {len(files)} files, {sum(map(len, files.values())):,} chars')


if __name__ == '__main__':
    main(sys.argv[1])
