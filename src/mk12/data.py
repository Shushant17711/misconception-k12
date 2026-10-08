"""Data loading and the frozen label space (R1.1)."""
import json, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, ART = ROOT / 'data', ROOT / 'artefacts'


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f'{split}.csv')
    df['image_path'] = [str(DATA / 'images' / split / f) for f in df.image_file]
    return df


def labels() -> list[str]:
    """Fixed label order: sorted train codes + NONE last. Frozen to disk on first call."""
    f = ART / 'labels.json'
    if f.exists():
        return json.loads(f.read_text())
    codes = sorted(set(load('train').root_code) - {'NONE'}) + ['NONE']
    ART.mkdir(exist_ok=True); f.write_text(json.dumps(codes))
    return codes


def check() -> dict:
    out = {}
    for s in ('train', 'test'):
        d = load(s)
        out[s] = int((~d.image_path.map(os.path.exists)).sum())
    return out


if __name__ == '__main__' and sys.argv[1:] == ['check']:
    print('labels', len(labels()), 'missing images', check())
