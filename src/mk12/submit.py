"""probs -> submission.csv with validation (R2.1)."""
import sys
import numpy as np, pandas as pd
from .data import load, labels, ROOT
from .log import load_probs


def write(P: np.ndarray, name: str) -> str:
    L = labels(); te = load('test')
    top = np.argsort(-P, axis=1)[:, :3]
    sub = pd.DataFrame({'trace_id': te.trace_id, 'root_code': [' '.join(L[j] for j in r) for r in top]})
    validate(sub)
    out = ROOT / 'submissions' / f'{name}.csv'; out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    return str(out)


def validate(sub: pd.DataFrame) -> None:
    te = load('test'); valid = set(labels())
    assert list(sub.columns) == ['trace_id', 'root_code'], 'bad header'
    assert sub.trace_id.is_unique and set(sub.trace_id) == set(te.trace_id), 'trace_id mismatch'
    for r in sub.root_code:
        codes = r.split()
        assert 1 <= len(codes) <= 3 and all(c in valid for c in codes), f'bad row {r!r}'


if __name__ == '__main__':
    exp = sys.argv[1]
    print(write(load_probs(exp)[1], exp))
