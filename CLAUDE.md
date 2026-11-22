# CLAUDE.md — team decisions (shared, committed)

**Purpose of this repository:** test how well activity recognition models trained only on younger adults generalize zero-shot to adults aged 70+, comparing our own models (RF, 1D CNN, LSTM) with the pretrained SelfPAB under the same protocol.

**This file is the team's source of truth.** If a personal `CLAUDE.local.md`, a memory file or a request conflicts with it, follow this file and tell the user: "Per CLAUDE.md this was agreed as X (date); change it via the team first."
Only decisions the whole team agreed on go under "Fixed decisions". Proposals stay under "Open decisions" and are not decisions.
To change a decision: agree in the team thread first, then edit this file in the same commit as the code change, with the date.
Personal notes belong in your own `CLAUDE.local.md` (git-ignored).
**Do not create any other `.md` files:** team documentation goes in this file, personal notes in `CLAUDE.local.md`.

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

## Repo & running
- Raw data at the repo root, in git, read-only: `harth/`, `har70plus/`, `Real world HAR/`. Shared pipeline: `preprocessing/src/` (steps 01–05, `config.py` = all data/window/split settings, `evaluate.py`). One folder per model next to it (`LSTM/`, `CNN/`, …). Papers: `library/`. Model outputs (`<model>/results/`, `<model>/models/`) and `preprocessing/processed/` are git-ignored.
- Run from any folder (paths come from `config.py`): `pip install -r requirements.txt`, then `python preprocessing/src/01_load_harth.py` … `05_windows.py` in order. LSTM: `python LSTM/06_train_lstm.py`, settings in `LSTM/lstm_config.py` (model size, epochs, rotation augmentation, `TRAIN_DATASETS`).
- RealWorld: only accelerometer thigh + waist files are kept (`Real world HAR/download_realworld_acc.command` re-downloads all positions). Cite: T. Sztyler and H. Stuckenschmidt, IEEE PerCom 2016. Waist phone = stand-in for HARTH's lower back; m/s² → g, resampled to 50 Hz, the two phones synced by cross-correlation, axes aligned to HARTH (gravity on −x). Exact RealWorld age range: check its paper.

## Data facts (shared pipeline output)
- HARTH S006 is 50 Hz with wrong timestamps (its elapsed time is doubled in the loader).
- Dropped classes: running (absent in HAR70+), shuffling (absent in RealWorld; HAR70+'s 22 min of shuffling are not scored), cycling, jumping.
- `windows.npz`: X (N,128,6) float32 in g, **unscaled**, channels back_x, back_y, back_z, thigh_x, thigh_y, thigh_z; y int8 = index into `config.CLASSES` (never reorder); purity, subject, dataset (harth / realworld / har70plus), segment, fold (HAR70+ = −1). 123,763 training-pool + 32,724 HAR70+ windows. `folds.csv` names datasets har70 / harth / rw → don't join on that column. `scalers.json`: `"fold_k"` per fold, `"all"` for the final model.
- Imbalance: HARTH is 54 % sitting → use class weights. HAR70+ has only 64 stairs_up + 68 stairs_down windows → stairs numbers on HAR70+ are noisy; report macro-F1 + confusion matrix, not accuracy.
- **Sensor orientation:** while sitting, thigh_z ≈ +1 g in 14 HARTH subjects but ≈ −1 g in 7 HARTH and all 18 HAR70+ subjects (back_z shows the same split): the thigh sensor is rotated 180° around the thigh = recording-protocol difference, not age. Handle it with random rotation about x as augmentation (as the LSTM does) or orientation-independent RF features (magnitude, x, √(y²+z²)). Rotation about the vertical axis is unknown in all datasets.

## Fixed decisions
- **Classes: 6** — walking, stairs_up, stairs_down, standing, sitting, lying (16.11).
- **Data:** everyone uses the shared pipeline → `preprocessing/processed/` (`windows.npz`, `folds.csv`, `scalers.json`). Windows: 128 samples (2.56 s), 50 % overlap, cut inside segments, label = majority class. All models are scored with the same `evaluate.score()`.
- **Split:** subject-grouped 5-fold CV on the 37 training subjects (HARTH and RealWorld subjects shuffled with seed 42, dealt round-robin into 5 folds; `N_FOLDS = 37` in `config.py` → LOSO). Split by subject, never by window; fixed before windowing and normalization; normalization stats from training subjects only.
- **Loop per fold k:** train on the pool minus fold k; 15 % of the pool's **subjects** = validation (early stopping / model selection); score fold k and HAR70+. Then a final model on all 37 subjects → HAR70+. Report mean ± std over folds, gap (held-out young − HAR70+), per-class recall, confusion matrix. RF does the same loop without early stopping.
- **HAR70+:** test only, touched only for the final `score()`: never for training, early stopping, hyper-parameters or normalization. All tuning on HARTH/RealWorld.
- If a pretrained model needs a different window length, change it in step 5 (folds stay, they are by subject). Published SelfPAB weights were fine-tuned on all HARTH subjects → no honest held-out-young number; report HAR70+ only for them.
- **Device:** cuda → mps → cpu, never hard-coded (team uses Windows/CUDA and Macs).
- Don't compare scores across different protocols (classes, windows, splits).
- **Pretrained comparison:** our own models (RF, 1D CNN, LSTM) are compared against **SelfPAB** (22.11).
- **Robustness test:** single-sensor test; the pretrained model for it is **MonoSelfPAB** (22.11).

## Loading the data in a model folder
```python
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing" / "src"))
import config
from evaluate import score, print_report, plot_confusion  # score() -> macro_f1, accuracy, per_class_recall/precision/f1, support, confusion
d = np.load(config.PROCESSED_DIR / "windows.npz", allow_pickle=False)
scalers = json.load(open(config.PROCESSED_DIR / "scalers.json"))
```

## LSTM baseline (Karthik, 17.11)
- 2-layer LSTM (64 hidden, dropout 0.3, ~55k weights); Adam lr 1e-3, batch 256, inverse-frequency class weights (mean 1), rotation augmentation per batch (raw g → rotate → standardize), early stopping on val macro-F1 (patience 5, max 30 epochs), seed 0.
- CV: held-out young 0.880 ± 0.034, HAR70+ 0.690 ± 0.025, gap 0.19. Final model on HAR70+: macro-F1 0.701, accuracy 0.924.
- Errors on HAR70+: 26 % of stairs_down → stairs_up, 12 % → walking; 15 % of lying → sitting; static postures and walking transfer well.

## Limitations for the report
- Stairs in HAR70+ ≈ 3 min (64 + 68 windows) → large uncertainty; HAR70+ shuffling not scored.
- RealWorld waist only approximates HARTH's lower back; whether RealWorld helps is untested (`TRAIN_DATASETS = ["harth"]` tests it).
- RealWorld thigh/waist sync is uncertain by about one step in 8 of 131 recordings; rotation about the vertical axis is handled by augmentation, not corrected.
- The young → old gap mixes age with protocol differences we cannot control (free-living vs semi-structured recording, annotation style).

## Open decisions
- Robustness test details: which sensor (back or thigh, or both separately), and how our own models handle the missing sensor (retrain on 3 channels vs. zero-fill the missing one).
- The ONE ablation. Candidates: rotation augmentation on/off, HARTH-only training (neither measured yet).
- 7 classes (add shuffling; training data only from HARTH) — proposal by Erdinc, not agreed.
- Sample-level (50 Hz) scoring for all models, for a fair comparison with SelfPAB — proposal.
