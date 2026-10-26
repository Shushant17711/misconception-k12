"""Learned re-ranker over Claude's calibrated top-k plus same-question neighbour labels.

Per (row, candidate label) features: Claude v1/v2/calibrated probs, label counts among labelled rows of the
same question, among those with the same final answer (read from Claude's transcription), and whether the
final answer matches correct_answer. HistGradientBoosting binary model; candidates ranked by its score.

  python -m mk12.rerank            # 5-fold CV on train
  python -m mk12.rerank --write    # fit on all train, write submissions/rerank.csv
"""
import re, sys
from fractions import Fraction
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .data import load, labels
from .blend_claude import claude_matrix
from .claude_diag import load_preds
from .calib_ens import avg, apply
from .calib_claude import fit_T
from .log import load_probs
from .metric import map3_probs
from .submit import write

TAGS = ['v1-opus', 'v2-cands-opus']
NUM = r'-?\d+(?:\.\d+)?(?:/\d+)?'


def _num(s):
    s = s.replace('−', '-').replace(',', '.').replace(' ', '')
    try: return Fraction(s).limit_denominator(10**6)
    except (ValueError, ZeroDivisionError): return None


def final(tx):
    if not tx: return None
    lines = [l.split('|', 1)[-1] for l in tx.split('\n') if '|' in l]
    cand = [l for l in lines if '//' in l] or lines[-1:]
    if not cand: return None
    s = re.sub(r'\[.*?\]', '', cand[-1].split('//')[0]).replace('−', '-')
    nums = re.findall(NUM, s.replace(' ', ''))
    return _num(nums[-1]) if nums else None


def correct(ans):
    s = str(ans)
    s = re.sub(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = re.sub(r'[\\$a-zA-Z{}^ ]', '', s).replace('(', '').replace(')', '')
    nums = re.findall(NUM, s.replace('−', '-'))
    return _num(nums[0]) if len(nums) == 1 else None


def finals(df):
    P = [load_preds(t) for t in TAGS]
    out = []
    for t in df.trace_id:
        fs = [final(p.get(t, {}).get('transcription')) for p in P]
        out.append(next((f for f in fs if f is not None), None))
    return out


def feats(df, Cs, Ccal, pool, L, self_ex=False):
    """Rows x K x F feature tensor. pool: DataFrame of labelled rows (question_id, fa, y, topic)."""
    K = len(L); N = K - 1
    byq = {q: g for q, g in pool.groupby('question_id')}
    byt = {t: np.bincount(g.y, minlength=K) for t, g in pool.groupby('topic_name')}
    glob = np.bincount(pool.y, minlength=K) / len(pool)
    F = []
    for i, r in enumerate(df.itertuples()):
        g = byq.get(r.question_id)
        if g is not None and self_ex: g = g[g.trace_id != r.trace_id]
        qc = np.bincount(g.y, minlength=K) if g is not None else np.zeros(K)
        if g is not None and r.fa is not None:
            gf = g[np.array([f == r.fa for f in g.fa], bool)]; fc = np.bincount(gf.y.astype(int), minlength=K)
        else: fc = np.zeros(K)
        tc = byt.get(r.topic_name, np.zeros(K)).astype(float)
        if self_ex and r.topic_name in byt: tc = tc.copy(); tc[r.y] -= 1
        ok = -1 if (r.fa is None or r.ca is None) else int(r.fa == r.ca)
        rank = (-Ccal[i]).argsort().argsort()
        isn = (np.arange(K) == N).astype(float)
        F.append(np.column_stack([Cs[0][i], Cs[1][i], Ccal[i], rank, qc, qc / max(qc.sum(), 1), np.full(K, qc.sum()),
                                  fc, fc / max(fc.sum(), 1), np.full(K, fc.sum()), tc / max(tc.sum(), 1), glob,
                                  isn, np.full(K, ok), isn * ok]))
    return np.stack(F)


def prep(df, L):
    idx = {c: i for i, c in enumerate(L)}
    df = df.copy(); df['fa'] = finals(df); df['ca'] = [correct(a) for a in df.correct_answer]
    if 'root_code' in df: df['y'] = df.root_code.map(idx)
    return df


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def fit_rank(X, y, cand):
    """X rows x K x F; train on candidate cells only."""
    r, c = np.where(cand)
    t = (c == y[r]).astype(int)
    return model().fit(X[r, c], t)


def score(clf, X, cand):
    S = np.full(cand.shape, -1.0)
    r, c = np.where(cand); S[r, c] = clf.predict_proba(X[r, c])[:, 1]
    return S


def candidates(Ccal, X, k=12):
    cand = np.zeros(Ccal.shape, bool)
    top = (-Ccal).argsort(1)[:, :k]
    cand[np.arange(len(Ccal))[:, None], top] = True
    cand |= X[:, :, 4] > 0  # any label seen on this question
    return cand


if __name__ == '__main__':
    L = labels(); K = len(L); tr = prep(load('train'), L); y = tr.y.values
    prior, pte = load_probs('prior')
    Cs = [claude_matrix(tr, load_preds(t), L) for t in TAGS]
    C, n = avg(tr, TAGS, L); assert (n == 2).all()
    f = np.random.default_rng(0).integers(0, 5, len(tr))
    cal = np.zeros((len(tr), K))
    for k in range(5):
        T = fit_T(C[f != k], y[f != k], K, 30); cal[f == k] = apply(C[f == k], T, prior[f == k], 0.05)
    print('calib baseline', round(map3_probs(cal, y), 4))
    print('fa extracted', tr.fa.notna().mean().round(3), 'ca parsed', tr.ca.notna().mean().round(3))
    out = np.zeros((len(tr), K)) - 1
    for k in range(5):
        a, b = f != k, f == k
        pool = tr[a]
        Xa = feats(tr[a], [Cs[0][a], Cs[1][a]], cal[a], pool, L, self_ex=True)
        Xb = feats(tr[b], [Cs[0][b], Cs[1][b]], cal[b], pool, L)
        ca, cb = candidates(cal[a], Xa), candidates(cal[b], Xb)
        clf = fit_rank(Xa, y[a], ca); out[b] = score(clf, Xb, cb)
    print('rerank CV', round(map3_probs(out, y), 4), 'gold in cands', candidates(cal, feats(tr, Cs, cal, tr, L, True))[np.arange(len(tr)), y].mean().round(3))
    if '--write' in sys.argv:
        te = prep(load('test'), L)
        T = fit_T(C, y, K, 30)
        Ce, ne = avg(te, TAGS, L); has = ne > 0
        calte = pte.copy(); calte[has] = apply(Ce[has], T, pte[has], 0.05)
        Cte = [claude_matrix(te, load_preds(t), L) for t in TAGS]
        Cte = [np.where(np.isnan(c), np.nan_to_num(Ce, nan=0) if False else calte, c) for c in Cte]
        Xa = feats(tr, Cs, cal, tr, L, self_ex=True); Xt = feats(te, Cte, calte, tr, L)
        clf = fit_rank(Xa, y, candidates(cal, Xa))
        S = score(clf, Xt, candidates(calte, Xt))
        # rank-preserving probs: scored candidates first, calibrated probs break the rest
        out = np.where(S >= 0, 1 + S, calte * 1e-3)
        print(write(out / out.sum(1, keepdims=True), 'rerank'))
