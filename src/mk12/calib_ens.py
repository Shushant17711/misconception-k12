"""Calibrated ensemble of several Claude runs: average claude_matrix over tags, fit_T on train, apply to test.

  python -m mk12.calib_ens v1-opus v2-cands-opus            # 5-fold CV on train rows covered by all tags
  python -m mk12.calib_ens v1-opus v2-cands-opus --write    # also write submissions/calib_avg_<n>.csv
Test rows missing a run use the mean of the runs they do have; rows with none fall back to the prior.
"""
import sys, itertools
import numpy as np
from .data import load, labels
from .blend_claude import claude_matrix
from .claude_diag import load_preds
from .metric import map3_probs
from .log import load_probs
from .calib_claude import fit_T
from .submit import write


def avg(df, tags, L):
    Ms = np.stack([claude_matrix(df, load_preds(t), L) for t in tags])
    return np.nanmean(Ms, 0), (~np.isnan(Ms[:, :, 0])).sum(0)


def apply(C, T, prior, beta):
    o = C @ T; o = o / o.sum(1, keepdims=True)
    return (1 - beta) * o + beta * prior


if __name__ == '__main__':
    tags = [a for a in sys.argv[1:] if not a.startswith('--')]
    L = labels(); K = len(L); tr = load('train'); te = load('test')
    y = tr.root_code.map({c: i for i, c in enumerate(L)}).values
    prior, pte = load_probs('prior')
    C, n = avg(tr, tags, L); m = np.where(n == len(tags))[0]
    f = np.random.default_rng(0).integers(0, 5, len(m))
    res = []
    for a, b in itertools.product([3, 10, 30], [0, 0.05, 0.1]):
        out = np.zeros((len(m), K))
        for k in range(5):
            T = fit_T(C[m[f != k]], y[m[f != k]], K, a)
            out[f == k] = apply(C[m[f == k]], T, prior[m[f == k]], b)
        res.append((map3_probs(out, y[m]), a, b))
    s, a, b = max(res)
    print(f'{"+".join(tags)}: n={len(m)} raw {map3_probs(C[m], y[m]):.4f} calib {s:.4f} alpha={a} beta={b}')
    if '--write' in sys.argv:
        T = fit_T(C[m], y[m], K, a)
        Ce, ne = avg(te, tags, L); has = ne > 0
        print('test coverage by #runs', np.bincount(ne, minlength=len(tags) + 1).tolist())
        out = pte.copy(); out[has] = apply(Ce[has], T, pte[has], b)
        print(write(out, f'calib_avg_{len(tags)}runs'))
