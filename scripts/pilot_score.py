import json, pandas as pd
from mk12.metric import ap3
d = pd.read_csv('data/train.csv').set_index('trace_id')
out = json.load(open('artefacts/pilot/output.json'))
s = []
for o in out:
    y = d.loc[o['trace_id'], 'root_code']; a = ap3(o['top3'], y); s.append(a)
    print(f"{a:.2f}  gold={y:7s} pred={' '.join(o['top3'])}")
print('pilot MAP@3', round(sum(s) / len(s), 3), 'n', len(s))
