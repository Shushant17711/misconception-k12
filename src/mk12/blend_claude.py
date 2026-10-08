"""Blend Claude's top-3 (with self-reported p) and the question prior (design §6, R5).

  python -m mk12.blend_claude <tag>            # fit on train rows Claude has seen, report
  python -m mk12.blend_claude <tag> --write    # also write submissions/claude_<tag>.csv
"""
import sys, itertools
import numpy as np, pandas as pd
from .data import load, labels
from .log import load_probs
from .metric import map3_probs
from .claude_diag import load_preds
from .submit import write


def claude_matrix(df: pd.DataFrame, preds: dict, L: list[str], floor: float = 0.02) -> np.ndarray:
    idx = {c: i for i, c in enumerate(L)}
    M = np.full((len(df), len(L)), np.nan)
    for i, t in enumerate(df.trace_id):
        p = preds.get(t)
        if not p: continue
        row = np.full(len(L), 1e-4)
        for rank, item in enumerate(p['top3']):
            c, pr = (item if isinstance(item, (list, tuple)) else (item, None))
            if c not in idx: continue
            try: pr = float(pr)
            except (TypeError, ValueError): pr = [0.6, 0.25, 0.1][rank]
            row[idx[c]] = max(row[idx[c]], max(pr, floor))
        M[i] = row / row.sum()
    return M


def combine(C, prior, w, lin):
    out = prior.copy(); has = ~np.isnan(C[:, 0])
    if lin:
        out[has] = w * C[has] + (1 - w) * prior[has]
    else:
        out[has] = np.exp(w * np.log(C[has]) + (1 - w) * np.log(prior[has] + 1e-9))
    return out


def fit(tag: str):
    L = labels(); tr = load('train'); y = tr.root_code.map({c: i for i, c in enumerate(L)}).values
    preds = load_preds(tag); C = claude_matrix(tr, preds, L); prior = load_probs('prior')[0]
    m = ~np.isnan(C[:, 0])
    print(f'{tag}: n={m.sum()}  prior {map3_probs(prior[m], y[m]):.4f}  claude {map3_probs(C[m], y[m]):.4f}')
    best = max(((map3_probs(combine(C, prior, w, lin)[m], y[m]), w, lin)
                for w, lin in itertools.product(np.linspace(0, 1, 21), [True, False])), key=lambda t: t[0])
    print(f'best blend {best[0]:.4f} w={best[1]:.2f} {"linear" if best[2] else "loglinear"}')
    return best


if __name__ == '__main__':
    tag = sys.argv[1]; s, w, lin = fit(tag)
    if '--write' in sys.argv:
        L = labels(); te = load('test'); C = claude_matrix(te, load_preds(tag), L)
        print('test coverage', (~np.isnan(C[:, 0])).mean().round(3))
        print(write(combine(C, load_probs('prior')[1], w, lin), f'claude_{tag}'))
