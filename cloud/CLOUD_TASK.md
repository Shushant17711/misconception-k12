# Cloud task: diagnose remaining test photos

You are working in a Claude Code cloud session. Do NOT install anything, do NOT run `claude -p`, and do NOT call any external API.
You yourself read the photos with the Read tool.

## Job
For each folder `cloud/batches/bNNNN/` that has no matching `cloud/out/bNNNN.json` yet:

1. Read `cloud/prompt.md` once at the start of the session. It defines the task, the taxonomy and the output format.
   Wherever it says "items.json in the current directory", that means `cloud/batches/bNNNN/items.json`.
2. Read that batch's `items.json`, then Read every image it names (`image` and `question_image`) from the same folder.
3. Follow the prompt for every item in the batch. Write ONLY the JSON array described in the prompt to
   `cloud/out/bNNNN.json`, one object per item with keys `id`, `transcription`, `reasoning`, `top3`.
   `top3` is `[["CODE", p], ["CODE", p], ["CODE", p]]`, using taxonomy ids or `NONE`.
4. After every 10 batches, `git add cloud/out && git commit -m "cloud out up to bNNNN" && git push`.

Work through batches in order, one at a time. Do not change anything outside `cloud/out/`.
At the end, commit and push, then report how many batches were written.
