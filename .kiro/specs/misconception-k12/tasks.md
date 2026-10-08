# Implementation Plan — Misconception Detection in K-12

Strategy: get a validated harness and a prior-only submission first, then the two-stage pipeline
(transcribe → classify), then blending and pseudo-labels. "Done" = one command rebuilds the best
OOF-validated submission. Kaggle uploads always wait for user confirmation (R2.3).

## Out-of-band prerequisites
- User decision: local VLM only, or hosted VLM API allowed for transcription (design §3)?
- User confirms each Kaggle submission.

## Task list

- [x] 1. Scaffold the package and shared contracts
  - `pyproject.toml` (uv), `src/mk12/`, MIT LICENSE, `.gitignore` for data/artefacts, `git init`
  - `data.py`: load train/test/unlabelled, image paths, `LABELS` fixed list saved to `artefacts/labels.json`
  - `data.py check` reports missing image files per split
  - _Requirements: 1.1, 7.2_

- [x] 2. Build the evaluation harness
- [x] 2.1 Implement MAP@3 metric with tests
  - `metric.py: map3(preds, labels)` with duplicate-skip and unknown-code-ignore
  - pytest: the three Evaluation-tab examples score 1.0; rank 2/3 → 0.5/0.333
  - _Requirements: 1.2_
- [x] 2.2 Freeze folds and the results log
  - `folds.py`: 5-fold StratifiedKFold on root_code (rare classes merged), seed 42 → `artefacts/folds.parquet`
  - `log.py: record(exp, oof_probs, config)` appends to `results.csv` with git hash; saves OOF/test arrays
  - _Requirements: 1.3, 1.4_

- [x] 3. Implement the question prior and first submission
  - `priors.py`: Dirichlet-smoothed question → topic → global prior, fitted in-fold; OOF + test probs
  - `submit.py`: probs → top-3 → `submission.csv`, validator (all trace_ids, ≤3 valid codes, header)
  - Test: prior OOF ≥ 0.45; validator rejects a broken file
  - _Requirements: 2.1, 2.2, 4.3_

- [ ] 4. Add a text baseline on gold transcriptions
  - `features.py: build_text(row, transcript)` shared prompt/input format for all classifiers
  - TF-IDF + logistic regression OOF on gold transcripts; blend with prior; log both
  - Establishes the ceiling gap between gold text and priors
  - _Requirements: 4.1, 1.4_

- [ ] 5. Build the transcriber
- [ ] 5.1 Implement the transcriber interface and cache
  - `transcribe.py`: `Transcriber.run(rows) -> {trace_id: text}`, disk cache keyed by model+prompt hash, resumable
  - Few-shot prompt with question text + 3 train examples in `L1 | …` format
  - _Requirements: 3.1, 3.3_
- [ ] 5.2 Add the local VLM backend and CER evaluation
  - 4-bit Qwen2.5-VL backend (7B AWQ, 3B fallback) within 8 GB
  - `transcribe eval`: CER vs gold on train images (per fold held-out); log throughput
  - _Requirements: 3.2_
- [ ] 5.3 Fine-tune the transcriber with LoRA if zero-shot CER is poor
  - LoRA on image → gold transcription, fold-wise so train transcripts stay OOF
  - Re-run CER eval; keep the better backend
  - _Requirements: 3.2_
- [ ] 5.4 Transcribe all train (OOF), test, and an unlabelled sample
  - Produce `artefacts/transcripts/<model>/` for 1,500 + 3,498 rows; then unlabelled in background
  - Re-score task 4 baseline on predicted transcripts (downstream metric for R3.2)
  - _Requirements: 3.1, 3.2_

- [ ] 6. Train the LLM classifier
- [ ] 6.1 Implement LoRA sequence classifier
  - `clf_llm.py`: Qwen ~1.5–4B, 4-bit, 76-way head + aux `final_answer_correct` head, 5-fold
  - Train on gold ∪ predicted transcripts; predict on predicted transcripts only
  - _Requirements: 4.1, 4.2, 5.2_
- [ ] 6.2 Score and log the classifier
  - OOF MAP@3 on predicted transcripts must beat the prior baseline; test probs averaged over folds
  - _Requirements: 4.4, 1.4_

- [ ] 7. Implement the blend and final ranking
  - `blend.py`: log-linear weights + temperature fitted on OOF MAP@3; NONE bias term from aux head
  - Writes blended test probs → `submit.py`; logs OOF of blend
  - _Requirements: 5.1, 5.2, 2.1_

- [ ] 8. Add pseudo-labelling on unlabelled data
  - Threshold sweep τ on blended confidence; retrain 6.1 with pseudo rows; keep only if OOF improves
  - _Requirements: 6.1_

- [ ] 9. Wire the one-command pipeline and README
  - `make submission` / `python -m mk12.pipeline` runs 3→7 from cached artefacts
  - README: approach, models + versions, hardware, runtime, cost, reproduce steps, MIT licence section
  - _Requirements: 7.1, 7.2_

## Milestones
- After 3: first valid submission (prior, ~0.45).
- After 5.4: know whether transcription is the bottleneck (gold vs predicted gap from task 4).
- After 7: competitive submission.

## Coverage
| Req | Tasks |
|---|---|
| 1.1 | 1 | 1.2 | 2.1 | 1.3 | 2.2 | 1.4 | 2.2, 4, 6.2 |
| 2.1 | 3, 7 | 2.2 | 3 | 2.3 | out-of-band |
| 3.1 | 5.1, 5.4 | 3.2 | 5.2–5.4 | 3.3 | 5.1 |
| 4.1 | 4, 6.1 | 4.2 | 6.1 | 4.3 | 3 | 4.4 | 6.2 |
| 5.1 | 7 | 5.2 | 6.1, 7 | 6.1 | 8 | 7.1 | 9 | 7.2 | 1, 9 |
