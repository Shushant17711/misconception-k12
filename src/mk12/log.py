"""Experiment log: OOF/test probs on disk + one results.csv row per run (R1.4)."""
import json, subprocess, time
import numpy as np, pandas as pd
from .data import ROOT, ART


def git_hash() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or 'nogit'
    except Exception:
        return 'nogit'


def record(exp: str, oof: np.ndarray, test: np.ndarray, score: float, config: dict) -> None:
    for kind, arr in (('oof', oof), ('test', test)):
        (ART / kind).mkdir(parents=True, exist_ok=True)
        np.save(ART / kind / f'{exp}.npy', arr.astype(np.float32))
    row = pd.DataFrame([{'time': time.strftime('%F %T'), 'exp': exp, 'oof_map3': round(score, 5),
                         'git': git_hash(), 'config': json.dumps(config)}])
    f = ROOT / 'results.csv'
    row.to_csv(f, mode='a', header=not f.exists(), index=False)
    print(f'[{exp}] OOF MAP@3 = {score:.4f}')


def load_probs(exp: str):
    return np.load(ART / 'oof' / f'{exp}.npy'), np.load(ART / 'test' / f'{exp}.npy')
