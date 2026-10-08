"""Resumable image fetcher: few workers, exponential backoff on 429."""
import os, sys, time, zipfile, random
from concurrent.futures import ThreadPoolExecutor
from kaggle.api.kaggle_api_extended import KaggleApi

C = 'misconception-detection-in-k-12-student-math-workings'
splits = sys.argv[1].split(',')
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
api = KaggleApi()
for a in range(12):
    try:
        api.authenticate(); break
    except BaseException as e:
        print('auth retry', a, e.__class__.__name__, flush=True); time.sleep(min(300, 15 * 2 ** a))
paths = [l.split(',')[0] for l in open('files.txt') if any(f'/images/{s}/' in l for s in splits)]
local = lambda p: 'data/' + p.removeprefix('competition_data/')
todo = [p for p in paths if not os.path.exists(local(p))]
print(f'{len(todo)} of {len(paths)} to fetch', flush=True)


def get(p):
    d = os.path.dirname(local(p)); os.makedirs(d, exist_ok=True)
    for a in range(8):
        try:
            api.competition_download_file(C, p, path=d, quiet=True)
            z = os.path.join(d, os.path.basename(p) + '.zip')
            if os.path.exists(z):
                zipfile.ZipFile(z).extractall(d); os.remove(z)
            if os.path.exists(local(p)):
                return True
        except BaseException:
            pass
        time.sleep(min(60, 2 ** a) + random.random())
    print('FAIL', p, flush=True)
    return False


n = fails = 0
with ThreadPoolExecutor(workers) as ex:
    for ok in ex.map(get, todo):
        n += 1; fails += not ok
        if n % 200 == 0:
            print(n, 'fails', fails, flush=True)
print('DONE fetched', n - fails, 'failed', fails, flush=True)
