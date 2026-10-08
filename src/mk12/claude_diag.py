"""Claude reader+diagnoser via headless `claude -p` (design §3, R3).

Each batch runs in an isolated temp dir that contains ONLY the batch's photos (renamed),
question diagrams, the taxonomy and items.json — never train.csv — with every tool except
Read disabled. Results are cached per trace_id under artefacts/claude/<tag>/.

  python -m mk12.claude_diag train 150      # first 150 train rows that have images
  python -m mk12.claude_diag test           # all test rows that have images
"""
import json, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from .data import load, DATA, ART

MODEL = os.environ.get('MK12_MODEL', 'opus')
TAG = os.environ.get('MK12_TAG', f'v1-{MODEL}')
BATCH = int(os.environ.get('MK12_BATCH', 8))
WORKERS = int(os.environ.get('MK12_WORKERS', 3))
OUT = ART / 'claude' / TAG
TAXONOMY = (DATA / 'taxonomy_v1.csv').read_text()

PROMPT = """You are diagnosing handwritten math working by Indonesian K-12 students (grades 4-12).

The misconception taxonomy (CSV) is:
<taxonomy>
{taxonomy}
</taxonomy>

items.json in the current directory lists {n} items: id, image (photo of the student's working, in this directory), grade, topic, question (Indonesian, LaTeX), question_image (diagram file or null), correct_answer (answer key).

For EVERY item: Read the photo (and question_image if present). Photos may be rotated, faint, or have crossed-out parts. Then
1. Transcribe the working line by line: "L1 | ...\\nL2 | ..." (verbatim; annotator notes in [brackets], e.g. [crossed out]).
2. Solve the question yourself and check each student step.
3. Pick the ROOT misconception: the major error (one that changes the reasoning or result) that the rest of the working follows from, not necessarily the first error written. If the working is fully correct or has only minor slips (e.g. imprecise notation that does not change the result) the label is NONE. About a third of items are NONE.
4. Give the 3 most likely codes (taxonomy ids or NONE), ranked, each with your probability (the three need not sum to 1).

Only read files in the current directory. Your FINAL message must be ONLY a JSON array, no prose, no code fence:
[{{"id": ..., "transcription": "...", "reasoning": "<=2 sentences", "top3": [["CODE", p], ["CODE", p], ["CODE", p]]}}, ...]
"""

DENY = ['Bash', 'Edit', 'Write', 'Glob', 'Grep', 'WebFetch', 'WebSearch', 'Agent', 'NotebookEdit', 'Skill', 'Task']


def run_batch(split: str, rows: pd.DataFrame) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory(prefix='mk12-') as tmp:
        tmp = Path(tmp); items = []
        for i, r in enumerate(rows.itertuples()):
            img = f'item{i}.jpeg'; shutil.copy(DATA / 'images' / split / r.image_file, tmp / img)
            qimg = None
            if isinstance(r.question_image_file, str) and (DATA / 'images/questions' / r.question_image_file).exists():
                qimg = f'q{i}_{r.question_image_file}'; shutil.copy(DATA / 'images/questions' / r.question_image_file, tmp / qimg)
            items.append(dict(id=f'item{i}', image=img, grade=int(r.grade), topic=r.topic_name,
                              question=r.question_text, question_image=qimg, correct_answer=str(r.correct_answer)))
        (tmp / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
        cmd = ['claude', '-p', PROMPT.format(taxonomy=TAXONOMY, n=len(items)), '--model', MODEL,
               '--output-format', 'json', '--no-session-persistence', '--strict-mcp-config',
               '--disallowedTools', *DENY]
        attempt = 0
        while attempt < 3:
            p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
            if 'hit your session limit' in p.stdout or 'usage limit' in p.stdout.lower():
                # subscription window exhausted: wait for reset without spending a retry
                print('usage limit; sleeping 10 min', time.strftime('%H:%M'), flush=True)
                time.sleep(600); continue
            attempt += 1
            try:
                meta = json.loads(p.stdout); txt = meta['result'].strip()
                txt = txt[txt.index('['): txt.rindex(']') + 1]
                res = {d['id']: d for d in json.loads(txt)}
                break
            except Exception as e:
                print('batch retry', attempt, type(e).__name__, (p.stdout or p.stderr)[-300:].replace('\n', ' '), flush=True)
                time.sleep(30 * attempt)
        else:
            return 0, {}
    n = 0
    for i, r in enumerate(rows.itertuples()):
        d = res.get(f'item{i}')
        if d and d.get('top3'):
            d['id'] = r.trace_id
            (OUT / f'{r.trace_id}.json').write_text(json.dumps(d, ensure_ascii=False)); n += 1
    usage = {k: meta.get(k) for k in ('total_cost_usd', 'duration_ms', 'num_turns')}
    usage.update(meta.get('usage', {}) and {k: meta['usage'].get(k) for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')})
    with open(OUT / '_usage.jsonl', 'a') as f:
        f.write(json.dumps({'split': split, 'n': len(rows), **usage}) + '\n')
    return n, usage


def todo(split: str, limit: int | None = None) -> pd.DataFrame:
    d = load(split)
    d = d[d.image_path.map(os.path.exists) & ~d.trace_id.map(lambda t: (OUT / f'{t}.json').exists())]
    return d.head(limit) if limit else d


def main(split: str, limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    d = todo(split, limit)
    batches = [d.iloc[i:i + BATCH] for i in range(0, len(d), BATCH)]
    print(f'{TAG}: {len(d)} {split} rows in {len(batches)} batches', flush=True)
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for n, u in ex.map(lambda b: run_batch(split, b), batches):
            done += n; print(f'done {done}/{len(d)} {u}', flush=True)


def load_preds(tag: str = TAG) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in (ART / 'claude' / tag).glob('*.json') if not p.stem.startswith('_')}


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
