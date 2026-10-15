"""Package the remaining v2-cands-opus test rows for a Claude Code cloud session, and import its answers.

  python -m mk12.cloud_export export   # writes cloud/ (images, items, prompt) for rows not yet cached
  python -m mk12.cloud_export import   # copies cloud/out/*.json into artefacts/claude/v2-cands-opus/

The cloud session itself reads the photos (no nested `claude -p`), so the work is billed to the
session's cloud credit. Answers use the same format as claude_diag: {"id", "transcription", "reasoning", "top3"}.
"""
import os, sys, json, shutil
os.environ.setdefault('MK12_TAG', 'v2-cands-opus'); os.environ.setdefault('MK12_CANDS', '6')
from pathlib import Path
from . import claude_diag as c
from .data import DATA

ROOT = Path(__file__).resolve().parents[2] / 'cloud'
BATCH = 8


def export():
    d = c.todo('test')
    if ROOT.exists(): shutil.rmtree(ROOT / 'batches', ignore_errors=True)
    (ROOT / 'out').mkdir(parents=True, exist_ok=True)
    (ROOT / 'out' / '.gitkeep').touch()
    for b, i0 in enumerate(range(0, len(d), BATCH)):
        bd = ROOT / 'batches' / f'b{b:04d}'; bd.mkdir(parents=True)
        items = []
        for i, r in enumerate(d.iloc[i0:i0 + BATCH].itertuples()):
            img = f'{r.trace_id}.jpeg'; shutil.copy(r.image_path, bd / img)
            qimg = None
            if isinstance(r.question_image_file, str) and (DATA / 'images' / 'questions' / r.question_image_file).exists():
                qimg = f'q_{r.trace_id}_{r.question_image_file}'
                shutil.copy(DATA / 'images' / 'questions' / r.question_image_file, bd / qimg)
            items.append(dict(id=r.trace_id, image=img, grade=int(r.grade), topic=r.topic_name,
                              question=r.question_text, question_image=qimg, correct_answer=str(r.correct_answer),
                              past_labels=c.past_labels('test', r.trace_id)))
        (bd / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
    prompt = c.PROMPT.format(taxonomy=c.TAXONOMY, n='the', cands_desc=c.CANDS_DESC)
    (ROOT / 'prompt.md').write_text(prompt)
    print(f'{len(d)} rows in {len(list((ROOT / "batches").iterdir()))} batches -> {ROOT}')


def do_import():
    n = 0
    for p in (ROOT / 'out').glob('b*.json'):
        for item in json.loads(p.read_text()):
            if item.get('top3'):
                (c.OUT / f'{item["id"]}.json').write_text(json.dumps(item, ensure_ascii=False)); n += 1
    print(f'imported {n} rows into {c.OUT}')


if __name__ == '__main__':
    export() if sys.argv[1] == 'export' else do_import()
