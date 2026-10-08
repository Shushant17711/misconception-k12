#!/usr/bin/env bash
# Download competition images listed in files.txt for the given split dirs.
set -u
C=misconception-detection-in-k-12-student-math-workings
cd "$(dirname "$0")"
grep -E "competition_data/images/($1)/" files.txt | cut -d, -f1 | while read -r p; do
  d="data/${p#competition_data/}"; [ -s "$d" ] || echo "$p"
done | xargs -P 16 -I{} sh -c 'p="{}"; d="data/${p#competition_data/}"; mkdir -p "$(dirname "$d")"; .venv/bin/kaggle competitions download '"$C"' -f "$p" -p "$(dirname "$d")" -q >/dev/null 2>&1; f="$(dirname "$d")/$(basename "$p")"; [ -f "$f.zip" ] && (cd "$(dirname "$d")" && unzip -oq "$(basename "$p").zip" && rm "$(basename "$p").zip"); true'
echo DONE; for s in questions train test; do echo "$s $(ls data/images/$s 2>/dev/null | wc -l)"; done
