"""Confusion-matrix recalibration of Claude's top-3 (tasks: second pass).

P_final = normalise(C @ T) where T[i, j] = smoothed P(gold = j | Claude names i), fitted on train
rows Claude has seen. alpha pulls T toward the identity; beta mixes the question prior for tie-breaks.

  python -m mk12.calib_claude v1-opus            # 5-fold CV on train rows with Claude preds
  python -m mk12.calib_claude v1-opus --write    # fit on all, write submissions/calib_<tag>.csv
"""
import sys, itertools
import numpy as np
from .data import load, labels
from .blend_claude import claude_matrix
from .claude_diag import load_preds
from .metric import map3_probs
from .log import load_probs
from .submit import write

LOG = '--log' in sys.argv  # log-linear prior mix (CV 0.8148 vs 0.8128 linear, n=1500)


def fit_T(C, y, K, alpha):
    T = alpha * np.eye(K)
    for row, g in zip(C, y):
        T[:, g] += row  # soft counts: Claude's mass on i, gold g
    return T / T.sum(1, keepdims=True)


def apply(C, T, prior, beta):
    out = C @ T
    out = out / out.sum(1, keepdims=True)
    if LOG: return np.exp(np.log(out + 1e-9) + beta * np.log(prior + 1e-9))
    return (1 - beta) * out + beta * prior


def cv(tag):
    L = labels(); K = len(L); tr = load('train')
    y = tr.trace_id.map(dict(zip(tr.trace_id, tr.root_code.map({c: i for i, c in enumerate(L)})))).values
    C = claude_matrix(tr, load_preds(tag), L); prior = load_probs('prior')[0]
    m = np.where(~np.isnan(C[:, 0]))[0]
    rng = np.random.default_rng(0); f = rng.integers(0, 5, len(m))
    print(f'{tag}: n={len(m)} raw {map3_probs(C[m], y[m]):.4f}')
    res = []
    for alpha, beta in itertools.product([1, 3, 10, 30, 100], [0, 0.05, 0.1]):
        out = np.zeros((len(m), K))
        for k in range(5):
            T = fit_T(C[m[f != k]], y[m[f != k]], K, alpha)
            out[f == k] = apply(C[m[f == k]], T, prior[m[f == k]], beta)
        res.append((map3_probs(out, y[m]), alpha, beta))
    for r in sorted(res, reverse=True)[:5]: print(f'  {r[0]:.4f} alpha={r[1]} beta={r[2]}')
    return max(res)


if __name__ == '__main__':
    tag = sys.argv[1]; s, alpha, beta = cv(tag)
    if '--write' in sys.argv:
        L = labels(); K = len(L); tr = load('train'); te = load('test')
        y = tr.root_code.map({c: i for i, c in enumerate(L)}).values
        P = load_preds(tag); Ct = claude_matrix(tr, P, L); m = ~np.isnan(Ct[:, 0])
        T = fit_T(Ct[m], y[m], K, alpha)
        Ce = claude_matrix(te, P, L); pte = load_probs('prior')[1]; has = ~np.isnan(Ce[:, 0])
        out = pte.copy(); out[has] = apply(Ce[has], T, pte[has], beta)
        print(write(out, f'calib_{tag}' + ('_log' if LOG else '')))
