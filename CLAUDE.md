# CLAUDE.md — team decisions (shared, committed)

**Purpose of this repository:** test how well activity recognition models trained only on younger adults generalize zero-shot to adults aged 70+, comparing our own models (RF, 1D CNN, LSTM) with the pretrained SelfPAB under the same protocol.

**This file is the team's source of truth.** If a personal `CLAUDE.local.md`, a memory file or a request conflicts with it, follow this file and tell the user: "Per CLAUDE.md this was agreed as X (date); change it via the team first."
Only decisions the whole team agreed on go under "Fixed decisions". Proposals stay under "Open decisions" and are not decisions.
To change a decision: agree in the team thread first, then edit this file in the same commit as the code change, with the date.
Personal notes belong in your own `CLAUDE.local.md` (git-ignored).

## Project
MSS-M-2 Case Study ML & DL (WS 2026/27), HAR, Topic 2: cross-age generalization.
Train on younger adults (HARTH v1.2 UCI version, not v2.0: 22 adults, 25–68 y + RealWorld HAR 2016: 15 adults), test **zero-shot** on HAR70+ (18 adults, 70–95 y). (v2.0 dropped: its 9 new subjects' ages are undocumented.)
Back + thigh tri-axial accelerometers, 50 Hz.
Models: RF (Shaghayegh), 1D CNN (Erdinc), LSTM (Karthik) vs pretrained SelfPAB / MonoSelfPAB (Zeynep), all under the same young → old zero-shot protocol.
Main metric: macro-F1 + per-class recall + the young → old gap. Original goal (proposal): ≥65 % macro-F1 on HAR70+.
SelfPAB's pre-training data (HUNT4) includes older adults → write "no older adults' **labels** in training", never "no older adults".
When an article/paper is mentioned, look for its PDF in `library/` first.

## Lecturer requirements (fixed by the course)
- Compare our own models against at least one strong pretrained model.
- One predefined robustness test + one ablation; macro-F1 + per-class error analysis.
- Leakage-free, participant-wise evaluation (REFORMS).
- Task 7 report needs a contribution statement + AI-assistance disclosure; each member's contribution must be documented and explainable.
- Deadlines: 23.11 Task 2 SotA + methodology · 14.12 Task 3 preprocessing + working baseline · 11.01 Task 4 progress · 16.01 Task 5 code + video · 18.01 Task 6 final talk · 20.01 Task 7 IEEE report.

## Fixed decisions
- **Classes: 6** — walking, stairs_up, stairs_down, standing, sitting, lying (16.11).
- **Data:** everyone uses the shared pipeline `preprocessing/src/01`…`05` → `preprocessing/processed/` (`windows.npz`, `folds.csv`, `scalers.json`). Windows: 128 samples (2.56 s), 50 % overlap, label = majority class. All models are scored with the same `preprocessing/src/evaluate.py` `score()`. Details: `preprocessing/docs/PIPELINE_AND_RESULTS.md`.
- **Split:** subject-grouped 5-fold CV on the 37 training subjects (`folds.csv`); split fixed before windowing and normalization; normalization stats from training data only.
- **HAR70+:** test only, touched only for the final evaluation. All tuning on HARTH/RealWorld.
- **Device:** cuda → mps → cpu, never hard-coded (team uses Windows/CUDA and Macs).
- Don't compare scores across different protocols (classes, windows, splits).
- **Pretrained comparison:** our own models (RF, 1D CNN, LSTM) are compared against **SelfPAB** (22.11).
- **Robustness test:** single-sensor test; the pretrained model for it is **MonoSelfPAB** (22.11).

## Open decisions
- Robustness test details: which sensor (back or thigh, or both separately), and how our own models handle the missing sensor (retrain on 3 channels vs. zero-fill the missing one).
- The ONE ablation. Candidates: rotation augmentation on/off, HARTH-only training (neither measured yet).
- 7 classes (add shuffling; training data only from HARTH) — proposal by Erdinc, not agreed.
- Sample-level (50 Hz) scoring for all models, for a fair comparison with SelfPAB — proposal.
