# Requirements — Misconception Detection in K-12 Student Math Workings

Competition: kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings
Deadline 2026-11-17 · 5 submissions/day · CSV upload (not a code competition) · MAP@3 over 75 taxonomy codes + `NONE`.

## Facts this spec rests on (verified 2026-10-08)
- train.csv 1,500 rows: question text (Bahasa Indonesia, LaTeX), correct answer, photo of handwritten working,
  line-by-line `transcription_steps`, `misconceptions` JSON (step, span, category, severity, root, text), `root_code`.
- test.csv 3,498 rows and unlabelled.csv 50,000 rows: same metadata + photo, **no transcription, no labels**.
- 95.7 % of test rows (97.6 % of unlabelled) share a `question_id` with train; ~3.2 train rows per question.
- `NONE` is ~1/3 of rows. Label is the *root* major misconception, not the first error.
- Baselines (LOO on train): global prior MAP@3 0.369, per-question prior 0.452. LB leader 0.672 on day 1.
- Hardware: RTX 4060 Laptop, 8 GB VRAM. External data/models allowed if "reasonably accessible" and "minimal cost".

## R1 Data & evaluation harness
- 1.1 All CSVs and images for train/test/questions are downloaded and indexed locally; missing files are reported.
- 1.2 A single `map3(preds, labels)` implementation matches the Evaluation-tab definition (duplicates skipped, ≤3 codes, unknown codes ignored).
- 1.3 Cross-validation is a fixed 5-fold stratified split on train, saved to disk, used by every model so OOF scores are comparable.
- 1.4 Every experiment logs OOF MAP@3, config, and git hash to one results table.

## R2 Submissions
- 2.1 A writer produces `submission.csv` with every test `trace_id`, ≤3 valid codes each, header `trace_id,root_code`, and validates it before saving.
- 2.2 A baseline submission (priors only) exists before any model work.
- 2.3 No submission is uploaded to Kaggle without explicit user confirmation.

## R3 Transcription of handwriting
- 3.1 Test/unlabelled images are transcribed into the same `L1 | … ` step format as train `transcription_steps`.
- 3.2 Transcription quality is measured on train images against gold transcriptions (character error rate + downstream OOF MAP@3 when the classifier is fed predicted vs gold text).
- 3.3 Transcriptions are cached on disk keyed by `trace_id` + transcriber version; reruns are incremental.

## R4 Misconception classifier
- 4.1 A model maps (question, correct answer, transcription, optional image) → ranked distribution over 76 labels.
- 4.2 Trained on gold transcriptions *and* predicted transcriptions so train/test inputs match.
- 4.3 Uses per-question priors (question_id → label history) as a feature or blend component.
- 4.4 Beats the per-question prior baseline on OOF MAP@3 computed with predicted (not gold) transcriptions.

## R5 Ranking & ensembling
- 5.1 Final top-3 is chosen to maximise expected MAP@3 from calibrated probabilities (blend weights / temperature fitted on OOF).
- 5.2 `NONE` handling is explicit: an `is_correct_working` signal can promote/demote NONE.

## R6 Semi-supervised use of unlabelled data
- 6.1 High-confidence pseudo-labels on unlabelled (and test) rows can be added to training, gated by an OOF improvement.

## R7 Reproducibility (prize eligibility)
- 7.1 One command regenerates the final submission from raw data + cached artefacts; README lists models, versions, hardware, cost.
- 7.2 Repo has an MIT LICENSE.
