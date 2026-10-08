import sys, json, numpy as np, pandas as pd
from mk12.claude_diag import load_preds
from mk12.metric import ap3
tag = sys.argv[1]
P = load_preds(tag); d = pd.read_csv('data/train.csv').set_index('trace_id')
s = [ap3([c for c, _ in p['top3']], d.loc[t, 'root_code']) for t, p in P.items() if t in d.index]
print(tag, 'n', len(s), 'MAP@3', round(np.mean(s), 3), 'top1', round(np.mean([x == 1 for x in s]), 3))
