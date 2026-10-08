# Design — Misconception Detection in K-12

## §1 Approach
Two-stage pipeline, because test has only photos while the labels are about reasoning steps:

```
image ──► [S1 Transcriber VLM] ──► L1..Ln text ──► [S2 Classifier] ──► p(76 labels) ─┐
question_id ──► [S3 Question prior] ─────────────────────────────────────────────────┼─► [S4 Blend + top-3] ─► submission.csv
image + question ──► [S2b direct VLM classifier, optional] ──────────────────────────┘
```

## §2 Repo layout
```
src/mk12/  data.py  metric.py  folds.py  priors.py  transcribe.py  features.py
           clf_llm.py  blend.py  submit.py  log.py
scripts/   one CLI per stage (python -m mk12.<stage>)
artefacts/ folds.parquet  transcripts/<model>/<trace_id>.txt  oof/<exp>.npy  test/<exp>.npy
results.csv  (exp, oof_map3, config, git)
```
Label space: `LABELS = taxonomy codes from train (74 seen) ∪ taxonomy_v1 if obtainable ∪ NONE`, fixed order, saved once.

## §3 S1 Transcriber
- Default: local open VLM that fits 8 GB in 4-bit (Qwen2.5-VL-7B-Instruct AWQ, fallback Qwen2.5-VL-3B), prompted with the question text and 3 train few-shot examples of the `L1 | …` format.
- Fine-tune option: LoRA on 1,500 (image → gold transcription) pairs if zero-shot CER is poor.
- Evaluated per R3.2 on 5-fold OOF train images. Cached per R3.3.
- Throughput target: 3,498 test images in < 6 h on the 4060.
- **Open decision (user):** a hosted frontier VLM (e.g. Claude/Gemini API) likely transcribes Indonesian handwriting far better, but costs money and must satisfy the "minimal cost" rule. Design keeps the transcriber behind one interface so either can be plugged in.

## §4 S2 Classifier
- Text-only LLM classifier: Qwen2.5/Qwen3 ~1.5–4B with LoRA + classification head (76-way), input = topic + question + correct answer + transcription. Fits 8 GB with 4-bit + gradient checkpointing.
- Training data per fold: gold transcriptions ∪ S1 predicted transcriptions of the same train images (augmentation; R4.2).
- Auxiliary target: `final_answer_correct` / `clean` (multi-task head) to drive NONE (R5.2).
- Cheap baseline before the LLM: TF-IDF char n-grams + logistic regression, to validate the harness.

## §5 S3 Question prior
Per-question label counts with Dirichlet smoothing toward topic-level and global priors; computed strictly within fold.

## §6 S4 Blend
Log-linear blend `log p = Σ w_i log p_i` with temperature, weights fitted on OOF MAP@3 (Nelder–Mead). Top-3 = argmax ordering of blended p (optimal for MAP@3 with a single true label).

## §7 Pseudo-labelling (R6)
Run S1+S2 on unlabelled; keep rows with blended max-prob > τ; retrain S2; accept only if OOF improves.

## §8 Acceptance criteria
1. `map3` unit tests reproduce the three Evaluation-tab examples (all 1.0).
2. Prior baseline OOF ≥ 0.45; its submission validates.
3. Transcriber CER reported on all 1,500 train images.
4. Classifier OOF on predicted transcripts > prior baseline OOF.
5. Final pipeline rebuilds `submission.csv` from one command.
