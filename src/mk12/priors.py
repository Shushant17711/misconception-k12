"""Hierarchical Dirichlet prior: question -> topic -> global (design §5, R4.3)."""
import numpy as np, pandas as pd
from .data import load, labels
from .folds import folds
from .metric import map3_probs
from .log import record


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, L: list[str], a_topic=2.0, a_q=1.0) -> np.ndarray:
    idx = {c: i for i, c in enumerate(L)}
    y = tr.root_code.map(idx).values
    def counts(keys):
        M = pd.crosstab(keys.values, y).reindex(columns=range(len(L)), fill_value=0)
        return M
    g = np.bincount(y, minlength=len(L)) + 0.5; g = g / g.sum()
    Ct, Cq = counts(tr.topic_name), counts(tr.question_id)
    out = np.zeros((len(te), len(L)))
    for i, (t, q) in enumerate(zip(te.topic_name, te.question_id)):
        pt = g * a_topic
        if t in Ct.index: pt = pt + Ct.loc[t].values
        pt = pt / pt.sum()
        pq = pt * a_q
        if q in Cq.index: pq = pq + Cq.loc[q].values
        out[i] = pq / pq.sum()
    return out


def run(a_topic=2.0, a_q=1.0, exp='prior'):
    L = labels(); tr, te = load('train'), load('test')
    f = folds().loc[tr.trace_id].values
    oof = np.zeros((len(tr), len(L)))
    for k in range(f.max() + 1):
        m = f == k
        oof[m] = fit_predict(tr[~m], tr[m], L, a_topic, a_q)
    y = tr.root_code.map({c: i for i, c in enumerate(L)}).values
    s = map3_probs(oof, y)
    test = fit_predict(tr, te, L, a_topic, a_q)
    record(exp, oof, test, s, {'a_topic': a_topic, 'a_q': a_q})
    return s


if __name__ == '__main__':
    import itertools
    L = labels(); tr = load('train'); f = folds().loc[tr.trace_id].values
    y = tr.root_code.map({c: i for i, c in enumerate(L)}).values
    best = None
    for at, aq in itertools.product([0.5, 1, 2, 5, 10], [0.25, 0.5, 1, 2, 4]):
        oof = np.zeros((len(tr), len(L)))
        for k in range(5):
            m = f == k; oof[m] = fit_predict(tr[~m], tr[m], L, at, aq)
        s = map3_probs(oof, y)
        if best is None or s > best[0]: best = (s, at, aq)
    print('best grid', best)
    run(best[1], best[2])
