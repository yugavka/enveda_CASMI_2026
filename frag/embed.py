"""Embed frag/frag_rescore.py into the notebook, in both places the pipeline writes /kaggle/working/frag_rescore.py.

1. the `_physical` string of the fragment-check cell (and the SHA-256 it asserts);
2. `FRAG_SRC` inside the pinned 0.465 pipeline (FROZEN_B64), re-encoding it and updating FROZEN_SHA256.

Both copies must stay identical: the pipeline overwrites the file the fragment check imports later.
Idempotent; re-run after every change to the scorer.

    .venv/bin/python -I frag/embed.py notebooks/casmi26-baseline-0477/casmi26-baseline-0477.ipynb
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

SCORER = Path(__file__).resolve().parent / 'frag_rescore.py'


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def find_cell(nb, needle):
    hits = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and needle in ''.join(c['source'])]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def splice_assign(src, name, value_src):
    """Replace the right-hand side of the unique assignment `name = ...` in `src` (searched at any depth)."""
    nodes = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == name for t in n.targets)]
    assert len(nodes) == 1, (name, len(nodes))
    v = nodes[0].value
    lines = src.splitlines(keepends=True)
    offs = [0]
    for line in lines:
        offs.append(offs[-1] + len(line.encode()))
    raw = src.encode()
    a = offs[v.lineno - 1] + v.col_offset
    b = offs[v.end_lineno - 1] + v.end_col_offset
    return (raw[:a] + value_src.encode() + raw[b:]).decode()


def set_source(cell, src):
    """Keep the cell's own layout: some cells of this notebook store their source as one string, not a list of lines."""
    cell['source'] = src if isinstance(cell['source'], str) else src.splitlines(keepends=True)


def main(path):
    path = Path(path)
    scorer = SCORER.read_text()
    nb = json.loads(path.read_text())

    # 1. fragment-check cell
    i = find_cell(nb, "_physical = ")
    src = ''.join(nb['cells'][i]['source'])
    old = ast.literal_eval(next(n.value for n in ast.parse(src).body
                                if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == '_physical'))
    src = splice_assign(src, '_physical', repr(scorer))
    assert src.count(sha(old)) == 1 or sha(old) == sha(scorer)
    src = src.replace(sha(old), sha(scorer))
    set_source(nb['cells'][i], src)

    # 2. pinned pipeline
    j = find_cell(nb, "FROZEN_B64 = ")
    src = ''.join(nb['cells'][j]['source'])
    tree = ast.parse(src)
    b64 = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'FROZEN_B64')
    frozen = json.loads(base64.b64decode(b64))
    k = find_cell(frozen, "FRAG_SRC = ")
    fsrc = splice_assign(''.join(frozen['cells'][k]['source']), 'FRAG_SRC', repr(scorer))
    set_source(frozen['cells'][k], fsrc)
    raw = json.dumps(frozen, separators=(',', ':')).encode()  # the pinned file's own format (ASCII, compact)
    src = splice_assign(src, 'FROZEN_B64', repr(base64.b64encode(raw).decode()))
    src = splice_assign(src, 'FROZEN_SHA256', repr(sha(raw)))
    set_source(nb['cells'][j], src)

    path.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')))  # the notebook's own format
    print(f'embedded {SCORER.name} sha256 {sha(scorer)[:12]} into cells {i} and {j} (pipeline cell {k}); '
          f'pipeline sha256 {sha(raw)[:12]}')


if __name__ == '__main__':
    main(sys.argv[1])
