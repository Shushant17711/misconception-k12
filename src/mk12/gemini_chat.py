"""Semi-manual Gemini (web chat) pass: package photos for upload, import pasted replies.

    python -m mk12.gemini_chat export test160   # 80 train rows where v1/v2 Opus disagree + 80 where they agree
    python -m mk12.gemini_chat import test160   # parse gemini/<set>/answers.txt -> artefacts/claude/v1-gemini/
    python -m mk12.gemini_chat eval test160     # MAP@3 vs Opus on the same rows

Each batch folder holds <=10 files (Gemini's per-message upload limit) and prompt.txt.
Every photo carries a black banner "ITEM gNNN" so file names don't matter after upload.
"""
import json, re, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .data import load, labels, DATA, ART
from . import claude_diag as c
from .claude_diag import load_preds
from .blend_claude import claude_matrix
from .metric import map3_probs

ROOT = Path(__file__).resolve().parents[2] / 'gemini'
FONT = '/usr/share/fonts/noto/NotoSans-Bold.ttf'
MAX_FILES, MAX_ITEMS, MAX_SIDE = 10, 8, 2400
TAG = 'v1-gemini'


def stamp(src: Path, dst: Path, text: str):
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    w, h = im.size; band = max(48, w // 18)
    out = Image.new('RGB', (w, h + band), 'black'); out.paste(im, (0, band))
    font = ImageFont.truetype(FONT, int(band * 0.7))
    ImageDraw.Draw(out).text((band // 4, band * 0.1), text, fill='white', font=font)
    out.save(dst, quality=90)


def prompt_for(items: list) -> str:
    p = c.PROMPT.format(taxonomy=c.TAXONOMY, n=len(items), cands_desc='')
    p = p.replace(
        "items.json in the current directory lists %d items: id, image (photo of the student's working, in this directory)" % len(items),
        "The %d items are listed below as JSON: id, has_question_image, grade, topic, question (Indonesian, LaTeX), correct_answer (answer key). "
        "The attached images each have a black banner at the top: 'ITEM <id>' is the photo of the student's working; "
        "'QUESTION <id>' is the question's diagram. Ignore the banner itself. Item fields: id, image (photo of the student's working)" % len(items))
    p = p.replace("Only read files in the current directory. Your FINAL message must be ONLY a JSON array, no prose, no code fence:",
                  "Answer every item. Your reply must be ONLY a JSON array, no other prose:")
    return p + '\n<items>\n' + json.dumps(items, ensure_ascii=False, indent=1) + '\n</items>\n'


def pick(name: str):
    tr = load('train'); P1, P2 = load_preds('v1-opus'), load_preds('v2-cands-opus')
    both = tr[tr.trace_id.isin(P1.keys()) & tr.trace_id.isin(P2.keys())]
    dis = both.trace_id.map(lambda t: P1[t]['top3'][0][0] != P2[t]['top3'][0][0])
    rng = np.random.default_rng(0)
    a = both[dis].sample(80, random_state=rng.integers(1e9)); b = both[~dis].sample(80, random_state=rng.integers(1e9))
    return tr.loc[rng.permutation(np.r_[a.index, b.index])], set(a.trace_id)


def export(name: str):
    rows, dis = pick(name); d = ROOT / name
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True); mp = {}; batches = []; cur = []; nf = 0
    for i, r in enumerate(rows.itertuples(), 1):
        gid = f'g{i:03d}'; mp[gid] = r.trace_id
        q = r.question_image_file if isinstance(r.question_image_file, str) and (DATA / 'images' / 'questions' / r.question_image_file).exists() else None
        k = 1 + (q is not None)
        if cur and (nf + k > MAX_FILES or len(cur) >= MAX_ITEMS):
            batches.append(cur); cur = []; nf = 0
        cur.append((gid, r, q)); nf += k
    batches.append(cur)
    for bi, b in enumerate(batches, 1):
        bd = d / f'b{bi:02d}'; bd.mkdir(); items = []
        for gid, r, q in b:
            stamp(Path(r.image_path), bd / f'{gid}.jpg', f'ITEM {gid}')
            if q: stamp(DATA / 'images' / 'questions' / q, bd / f'{gid}_question.jpg', f'QUESTION {gid}')
            items.append(dict(id=gid, has_question_image=q is not None, grade=int(r.grade), topic=r.topic_name,
                              question=r.question_text, correct_answer=str(r.correct_answer)))
        (bd / 'prompt.txt').write_text(prompt_for(items))
    (d / 'map.json').write_text(json.dumps(dict(map=mp, disagree=sorted(dis)), indent=1))
    (d / 'answers.txt').touch()
    print(f'{len(rows)} rows in {len(batches)} batches -> {d}')


def parse(text: str) -> list:
    dec = json.JSONDecoder(); out = []
    for m in re.finditer(r'\[\s*\{', text):
        try: v, _ = dec.raw_decode(text[m.start():])
        except ValueError: continue
        out += [x for x in v if isinstance(x, dict) and x.get('id') and x.get('top3')]
    return out


def do_import(name: str):
    d = ROOT / name; mp = json.loads((d / 'map.json').read_text())['map']
    got = {x['id']: x for x in parse((d / 'answers.txt').read_text())}
    out = ART / 'claude' / TAG; out.mkdir(parents=True, exist_ok=True); n = 0
    for gid, x in got.items():
        if gid in mp:
            x = dict(x, id=mp[gid]); (out / f'{mp[gid]}.json').write_text(json.dumps(x, ensure_ascii=False)); n += 1
    print(f'imported {n}/{len(mp)}; missing: {sorted(set(mp) - set(got))[:30]}')


def evaluate(name: str):
    d = ROOT / name; meta = json.loads((d / 'map.json').read_text())
    L = labels(); tr = load('train').set_index('trace_id'); G = load_preds(TAG)
    ids = [t for t in meta['map'].values() if t in G]; df = tr.loc[ids].reset_index()
    y = df.root_code.map({k: i for i, k in enumerate(L)}).values
    M = {t: claude_matrix(df, load_preds(t), L) for t in ['v1-opus', 'v2-cands-opus', TAG]}
    dis = df.trace_id.isin(meta['disagree']).values
    combos = {'opus v1': M['v1-opus'], 'opus v2': M['v2-cands-opus'], 'gemini': M[TAG],
              'opus avg': (M['v1-opus'] + M['v2-cands-opus']) / 2,
              'opus avg + gemini (1/3 each)': (M['v1-opus'] + M['v2-cands-opus'] + M[TAG]) / 3,
              'opus avg + gemini (50/50)': (M['v1-opus'] + M['v2-cands-opus']) / 4 + M[TAG] / 2}
    print(f'n={len(df)} (disagree {dis.sum()}, agree {(~dis).sum()})')
    print(f'{"":30s} {"all":>6s} {"disagree":>9s} {"agree":>6s}')
    for k, A in combos.items():
        f = lambda m: map3_probs(A[m], y[m]) if m.any() else float('nan')
        print(f'{k:30s} {f(np.ones_like(dis)):6.3f} {f(dis):9.3f} {f(~dis):6.3f}')


if __name__ == '__main__':
    {'export': export, 'import': do_import, 'eval': evaluate}[sys.argv[1]](sys.argv[2])
