"""Frozen 5-fold split (R1.3)."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .data import load, ART

K = 5


def folds() -> pd.Series:
    f = ART / 'folds.parquet'
    if f.exists():
        return pd.read_parquet(f).set_index('trace_id').fold
    d = load('train')
    vc = d.root_code.value_counts()
    strat = d.root_code.where(d.root_code.map(vc) >= K, 'RARE')
    fold = pd.Series(-1, index=d.index)
    for i, (_, va) in enumerate(StratifiedKFold(K, shuffle=True, random_state=42).split(d, strat)):
        fold.iloc[va] = i
    out = pd.DataFrame({'trace_id': d.trace_id, 'fold': fold.values})
    ART.mkdir(exist_ok=True); out.to_parquet(f)
    return out.set_index('trace_id').fold
