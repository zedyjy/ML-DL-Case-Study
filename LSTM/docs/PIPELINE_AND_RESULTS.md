# Cross-age HAR — data pipeline, evaluation protocol and LSTM baseline

*Karthik Yerragadda, 17 Nov 2026. Topic 2: "Evaluating the generalization of
pretrained HAR models across age groups".*

This document explains, step by step, how the raw datasets were turned into
model-ready windows, what was removed and why, which problems came up, how
the evaluation protocol works and what the LSTM baseline achieved. The goal
is that everyone in the group can (a) understand the numbers, and (b) run the
CNN, Random Forest, SelfPAB and MonoSelfPAB on **exactly the same windows,
folds and metric**, so that the five models are comparable.

All code is in `src/` (plain Python with `# %%` cell markers). Every number
below is printed by the scripts themselves.

---

## 1. The big picture

| | HARTH | RealWorld HAR 2016 | HAR70+ |
|---|---|---|---|
| role | **training** (younger adults) | **training** (younger adults) | **test only** (older adults) |
| subjects | 22, age 25–68 | 15, younger adults (check the dataset paper for the exact age range) | 18, age 70–95 |
| sensors | 2× Axivity AX3: lower back + right thigh | 2 smartphones: waist + thigh (pocket) | same as HARTH |
| units | g | m/s² → converted to g | g |
| rate | 50 Hz | ~49.95 Hz, irregular timestamps → resampled to 50 Hz | 50 Hz (±1 ms jitter) |
| labels | 12 codes, video-annotated | activity name in the file name | 7 codes |
| raw rows | 6,461,328 | 263 files (131 recordings) | 2,259,597 |

**The question:** train on younger adults, apply the model *unchanged* to
older adults, and measure how much performance is lost and on which
activities. HAR70+ is therefore never used for training, validation, or even
for computing normalisation statistics. This is called **zero-shot**
evaluation.

```
raw CSVs --> 01/02/03 load & clean --> 04 common labels --> 05 windows + folds + scaler --> 06 LSTM
              (one Parquet table         (one table,           windows.npz, folds.csv,         5-fold CV
               per dataset)               6 classes)            scalers.json                    + HAR70+
```

---

## 2. Pipeline, step by step

### Step 1 — HARTH (`src/01_load_harth.py`)

* Reads the 22 CSVs. Three header variants exist (S015/S021 have an extra
  `index` column, S023 an unnamed first column); the extra column is dropped.
* Timestamps are converted to *seconds since the file started*. The absolute
  dates are fake anyway (most files start at 2019-01-12 00:00:00 or
  2000-01-01).
* **Gap splitting.** Every file contains time jumps (0.5 s up to 302 s)
  where the recording was paused. A recording is cut into *segments* at every
  jump > 0.1 s, and later no window is allowed to cross a segment boundary.
  Result: 666 segments.
* Inside each segment the time axis is rebuilt as 0, 0.02, 0.04 … s and
  checked against the real timestamps (max deviation 0.8 ms → the data is a
  true 50 Hz).
* Output: `processed/harth.parquet` (6,461,328 rows — matches the raw count).

**Problem found: S006 looks like 100 Hz.** Its timestamps step by 10 ms while
every other file steps by 20 ms. Is it really 100 Hz, or a timestamp error?
Check: the walking cadence of S006 is 1.86 steps/s if read at 50 Hz (every
other subject: 1.5–2.1 steps/s), but 3.8 steps/s if read at 100 Hz — a sprint.
Also, read at 50 Hz its session is 139 min, like the other ~2 h sessions.
**Conclusion: S006 is 50 Hz with wrong timestamps.** Its elapsed time is
doubled in the loader.

### Step 2 — HAR70+ (`src/02_load_har70.py`)

* Same recipe as step 1. All 18 files have the clean 8-column header.
* Timestamps jitter between 19, 20 and 21 ms. Verified that the jitter does
  not accumulate (19s and 21s come in equal numbers; max deviation from a
  perfect 50 Hz grid is 1.0 ms), so the data is treated as 50 Hz.
* Real gaps of 3–112 s exist → 273 segments.
* Orientation check: while standing, all 18 subjects read ≈ −1 g on `back_x`
  and `thigh_x`, i.e. the same mounting as HARTH ("x = down").
* Output: `processed/har70plus.parquet` (2,259,597 rows).

### Step 3 — RealWorld (`src/03_load_realworld.py`) — the hard one

Each recording is **two files** (thigh phone, waist phone) with separate
clocks, in m/s², with a phone orientation that differs between subjects and
even between recordings. Only thigh and waist were kept of the seven body
positions; waist is our stand-in for HARTH's lower back.

What the script does per recording (subject × activity × recording number):

1. **Pairs the thigh and waist file**, converts m/s² → g (÷ 9.80665).
2. **Repairs the clock.** The timestamps have ~100 "holes" of 1–2 s per file,
   each followed by a burst of samples stamped 1 ms apart. These are *not*
   missing data: Android buffered the samples and stamped them late. The
   phones actually sampled at a rock-steady 49.952 Hz. So the script trusts
   the *sample count* and fits a straight line `time = a + b·index` through
   the timestamps. A few files contain one *real* dropout (a permanent jump of
   2–3 s, e.g. s04 climbingup waist, s13 lying waist); those are detected and
   the file is split there.
3. **Resamples both phones onto one shared 50 Hz grid** (linear
   interpolation) over their overlapping time range.
4. **Synchronises the two phones.** Their clocks disagree by up to 7 s, and
   the offset changes from recording to recording. For every moving recording
   the acceleration magnitudes of thigh and waist are cross-correlated (both
   phones feel every footstep) and the waist clock is shifted by the best lag.
   Quality check: the lag is re-measured in each quarter of the recording;
   stairs agree to ±0.05 s, walking is ambiguous by about one step (0.5–1 s)
   in 8 of 131 recordings — the best a periodic signal allows. Static
   recordings (standing, sitting, lying) use the subject's median lag; there
   the offset does not matter.
5. **Aligns the axes to the HARTH convention** (gravity on −x while upright).
   Per *recording*, the mean acceleration of an upright activity is taken as
   the "down" direction and the smallest rotation that turns it into (−1,0,0)
   is applied (Rodrigues formula). Sitting and lying have no "down" reference
   of their own (the thigh is horizontal — that *is* the information), so they
   reuse the subject's standing rotation. A waist that was upside-down
   relative to standing (s03 sitting) is flipped back. Why per recording and
   not per subject: for several subjects the phone moved between recordings
   (s13's thigh phone: 38° between standing and walking; s08's waist: 179° in
   climbing up).
6. Saves `processed/realworld.parquet` (3,373,819 rows, label = activity
   name) and `processed/realworld_recording_log.csv` (lag, lag spread and
   mean vector of every recording — use it to audit).

Skipped: s06 jumping (no thigh file).

**What cannot be fixed: rotation about the vertical axis.** After alignment
we know which way is down, but not which way the phone's y/z axes point
around the body axis. This is a general problem, not just RealWorld's — see
section 4.

### Step 4 — Common labels (`src/04_labels.py`, tables in `src/config.py`)

Only the six activities that all three datasets share are kept:

| common class (index) | HARTH / HAR70+ code | RealWorld file name |
|---|---|---|
| walking (0) | 1 | walking |
| stairs_up (1) | 4 | climbingup |
| stairs_down (2) | 5 | climbingdown |
| standing (3) | 6 | standing |
| sitting (4) | 7 | sitting |
| lying (5) | 8 | lying |

Dropped, and why:

| dropped | present in | reason |
|---|---|---|
| running (code 2) | HARTH, RealWorld | absent in HAR70+; keeping it would make the train and test class sets differ |
| shuffling (3) | HARTH, HAR70+ | absent in RealWorld; also the class the original HARTH paper's models confused most; keeping it would mix "can the LSTM learn shuffling" with "does it generalise across age". Its 22 min in HAR70+ are therefore not scored — a limitation to mention |
| cycling (13, 14, 130, 140) | HARTH only | — |
| jumping | RealWorld only | — |

Removed rows: HARTH 16.2 %, HAR70+ 2.9 % (shuffling only), RealWorld 18 %.
Dropping rows creates new holes inside segments (walking → *running* →
walking); segments are re-split at every hole (→ 4,151 segments; 942 are now
shorter than one window and yield nothing, 20 min of data in total).

Minutes per class after mapping:

| | HAR70+ (test) | HARTH | RealWorld |
|---|---|---|---|
| walking | 360 | 399 | 162 |
| stairs_up | **1.5** | 25 | 157 |
| stairs_down | **1.7** | 22 | 126 |
| standing | 139 | 248 | 158 |
| sitting | 161 | **968** | 160 |
| lying | 68 | 143 | 160 |

Two consequences: HARTH is 54 % sitting → class weights are needed in
training; HAR70+ has only ~3 minutes of stairs → stairs numbers on HAR70+ are
noisy, so we report macro-F1 **and** the confusion matrix, not accuracy.

### Step 5 — Windows, folds, scaler (`src/05_windows.py`)

* **Windows**: 128 samples (2.56 s at 50 Hz), sliding by 64 (50 % overlap),
  cut inside segments only. Label = majority class in the window; `purity`
  = share of that class (97.3 % of windows are pure). Each window is one
  example of shape (128, 6): 128 time steps × [back_x, back_y, back_z,
  thigh_x, thigh_y, thigh_z] in g.
* **Count**: 156,487 windows = 123,763 training pool (HARTH + RealWorld) +
  32,724 HAR70+.
* **Folds**: see section 3.
* **Scaler**: per-channel mean/std for standardisation, fitted on training
  subjects only — one per fold and one on all 37 training subjects. HAR70+
  never contributes. Stored in `processed/scalers.json`. The windows
  themselves are saved **unscaled** (in g) so every model can decide how to
  normalise, and so that the rotation augmentation can be applied before
  scaling.
* Output: `processed/windows.npz` (510 MB) with arrays `X (N,128,6) float32`,
  `y (N,) int8`, `purity`, `subject`, `dataset`, `segment`, `fold`, `classes`.

### Step 6 — LSTM (`src/06_train_lstm.py`) — see section 5.

---

## 3. Evaluation protocol (please use the same one)

### 3.1 What is compared

Two numbers per model:

1. **HAR70+ macro-F1 (zero-shot)** — *the* cross-model comparison. Every
   model (LSTM, CNN, RF, SelfPAB, MonoSelfPAB) is scored on the same 32,724
   HAR70+ windows, the same 6 classes, the same metric. It does not depend on
   how the younger adults are split.
2. **Generalization gap** = macro-F1 on *held-out younger adults* minus
   macro-F1 on HAR70+. For this the group must use the same held-out
   subjects, otherwise the gaps are not comparable. Note: the published
   SelfPAB weights were fine-tuned on **all** HARTH subjects, so the
   pretrained models cannot give an honest held-out-HARTH number; for them
   report the HAR70+ side only.

### 3.2 Why subject-grouped 5-fold cross-validation

Three options were considered for the younger-adult split:

| option | trainings per experiment | what you get | why (not) |
|---|---|---|---|
| one fixed held-out set (random subjects, fixed seed) | 1 | one number, no error bar | fastest, but a single number depends on which 7 subjects happened to be held out |
| leave-one-subject-out (LOSO) | 37 | every subject scored once, error bar | gold standard, but ~6–12 h per experiment on a laptop, and we want several experiments (±RealWorld, ±augmentation, model sizes) |
| **subject-grouped 5-fold CV (chosen)** | 5 | every subject scored once, error bar | same property as LOSO (each subject is tested by a model that never saw them), at 1/7 of the cost. Standard in HAR papers with 20+ subjects. Setting `N_FOLDS = 37` in `config.py` turns it into LOSO without any other change |

How the folds were made: the 22 HARTH and 15 RealWorld subjects were each
shuffled with seed 42 and dealt round-robin into 5 folds, so every fold has
4–5 HARTH + 3 RealWorld subjects. All HAR70+ subjects have fold −1.

| fold | subjects |
|---|---|
| 0 | S006, S021, S023, S025, S028, rw_s01, rw_s12, rw_s14 |
| 1 | S009, S015, S016, S018, S026, rw_s09, rw_s10, rw_s11 |
| 2 | S012, S013, S014, S024, rw_s04, rw_s08, rw_s15 |
| 3 | S017, S019, S027, S029, rw_s06, rw_s07, rw_s13 |
| 4 | S008, S010, S020, S022, rw_s02, rw_s03, rw_s05 |
| −1 | all HAR70+ (501–518) — test only |

(`processed/folds.csv`.)

The split is by **subject**, never by window: neighbouring windows overlap by
half, so windows of the same person in both train and test would be leakage.

### 3.3 The loop, for fold k = 0 … 4

1. training pool = HARTH + RealWorld windows **not** in fold k;
2. 15 % of the pool's *subjects* become the validation set (early stopping /
   model selection) — never the held-out fold itself, never HAR70+;
3. train;
4. score fold k (younger adults never seen) **and** HAR70+ with the same model.

Then train a final model on all 37 subjects and score HAR70+ once more.
Report: mean ± std over the 5 folds for both numbers, the gap, per-class
recall and the confusion matrix. A classical model (Random Forest) does the
same loop; it just has no early stopping.

### 3.4 Rules that keep the comparison fair

* HAR70+ windows are never used for anything except the final `score()` call —
  not for training, early stopping, hyper-parameter choice, or normalisation.
* Normalise with statistics from training subjects only (`scalers.json` has
  one entry per fold and `"all"` for the final model). If your model does its
  own normalisation (e.g. SelfPAB's preprocessing), fit it on the training
  folds only.
* Same windows, same classes, same `evaluate.score()`.
* If a pretrained model needs a different window length, say so — it is a
  config change in step 5, and the fold assignment stays identical because it
  is by subject.

### 3.5 How to load everything (copy-paste)

```python
import numpy as np, json, sys
sys.path.insert(0, "src")
import config
from evaluate import score, print_report, plot_confusion

d = np.load("processed/windows.npz", allow_pickle=False)
X, y, fold, dataset = d["X"], d["y"], d["fold"], d["dataset"]   # X: (N, 128, 6) in g
scalers = json.load(open("processed/scalers.json"))

is_test = dataset == "har70plus"                                   # fold == -1
for k in range(config.N_FOLDS):
    train = (~is_test) & (fold != k)
    heldout = (~is_test) & (fold == k)
    mean, std = np.array(scalers[f"fold_{k}"]["mean"]), np.array(scalers[f"fold_{k}"]["std"])
    Xtr, Xho, Xte = (X[train] - mean) / std, (X[heldout] - mean) / std, (X[is_test] - mean) / std
    # ... fit your model on (Xtr, y[train]) — split some subjects off for validation ...
    print_report(score(y[heldout], model.predict(Xho)), f"fold {k} held-out")
    print_report(score(y[is_test], model.predict(Xte)), f"fold {k} HAR70+")
```

`score()` returns a dict: `macro_f1`, `accuracy`, `per_class_recall`,
`per_class_precision`, `per_class_f1`, `support`, `confusion`. Class index
→ name: `config.CLASSES = ["walking", "stairs_up", "stairs_down",
"standing", "sitting", "lying"]`.

---

## 4. A finding that affects every model: sensor orientation

While checking the gravity direction per posture, this came out:

| sitting, mean `thigh_z` | HARTH | HAR70+ |
|---|---|---|
| ≈ +1.0 g | 14 subjects | 0 subjects |
| ≈ −1.0 g | 7 subjects | 18 subjects |

Both datasets have the thigh sensor's x axis pointing down, but the sensor is
rotated 180° around the thigh on *some* HARTH subjects and on *all* HAR70+
subjects. That is a **recording-protocol difference, not an age effect**
(risk 4 in our proposal). A model trained on HARTH would mostly learn
"sitting ⇒ thigh_z ≈ +1" and then see −1 on every older adult. The sign of
`back_z` while sitting shows the same split.

How the LSTM handles it: **random rotation about the x axis as training
augmentation** — each window's (y, z) of each sensor is rotated by a random
angle, so the model learns to ignore how the sensor is turned around the body
axis. This also covers RealWorld's unknown y/z orientation.

**For the CNN / RF / SelfPAB:** either use the same augmentation (for the RF:
compute orientation-independent features, e.g. magnitude, x, and
sqrt(y²+z²)), or be aware that part of the HARTH → HAR70+ drop may be this
mounting difference rather than age. An ablation with `AUGMENT_ROTATION =
False` would quantify it; I have not run it.

---

## 5. The LSTM baseline

**Model** (`LSTMClassifier` in `src/06_train_lstm.py`, PyTorch):
`nn.LSTM(6 → 64 hidden, 2 layers, dropout 0.3, batch_first)` reads the 128
time steps; the hidden state after the last step (64 numbers = summary of the
2.56 s) goes through dropout and a linear layer to 6 class scores. ≈ 55 k
weights.

**Training**: Adam, lr 1e-3, batch 256, cross-entropy with inverse-frequency
class weights (normalised to mean 1 — a stairs window weighs ≈ 50× a sitting
window), rotation augmentation on every batch, early stopping on validation
macro-F1 with patience 5 (best weights restored), max 30 epochs. Per batch:
raw g → rotate → standardise with the fold's scaler → model. Seed 0.
≈ 18 s per epoch on an Apple-silicon GPU; the full run (5 folds + final)
took ≈ 35 min.

**Results** (run `lstm_harth+realworld_rot`, `processed/results/`):

| fold | held-out younger adults | HAR70+ | gap |
|---|---|---|---|
| 0 | 0.850 | 0.707 | 0.14 |
| 1 | 0.927 | 0.697 | 0.23 |
| 2 | 0.862 | 0.716 | 0.15 |
| 3 | 0.856 | 0.656 | 0.20 |
| 4 | 0.905 | 0.672 | 0.23 |
| **mean ± std** | **0.880 ± 0.034** | **0.690 ± 0.025** | **0.19** |

Final model (all 37 training subjects) on HAR70+: **macro-F1 0.701**,
accuracy 0.924 (target in the proposal: ≥ 0.65).

Per-class recall on HAR70+, final model:

| class | recall | precision | support (windows) |
|---|---|---|---|
| walking | 0.90 | 0.99 | 16,638 |
| stairs_up | 0.73 | 0.05 | 64 |
| stairs_down | 0.53 | 0.32 | 68 |
| standing | 0.97 | 0.88 | 5,367 |
| sitting | 0.98 | 0.94 | 7,445 |
| lying | 0.85 | 0.94 | 3,142 |

What breaks for older adults (confusion matrix
`lstm_harth+realworld_rot_cm_har70.png`): 26 % of stairs_down is called
stairs_up and 12 % walking; 15 % of lying is called sitting; stairs precision
is tiny because a few percent of 16,638 walking windows called "stairs"
already outnumber the 64 real stairs windows. Static postures and walking
transfer well. Plausible explanation: older adults take stairs slowly and
carefully, which at the thigh looks like slow walking.

---

## 6. Limitations to state in the report

* Stairs in HAR70+: 64 + 68 windows (≈ 3 min). Any stairs number on HAR70+
  has a large uncertainty.
* HAR70+ shuffling (22 min) is not scored because the class was dropped.
* RealWorld's waist phone is only an approximation of HARTH's lower-back
  sensor; whether it helps is untested (`TRAIN_DATASETS = ["harth"]` would
  test it).
* RealWorld thigh/waist synchronisation is uncertain by about one step in 8
  of 131 recordings.
* Rotation about the vertical axis is unknown in all datasets; it is handled
  by augmentation, not corrected.
* The HARTH → HAR70+ gap mixes age with other protocol differences we could
  not control (free-living vs. semi-structured recording, annotation style).

---

## 7. Reproducing everything

Paths in this document (`src/`, `processed/`) are relative to the `LSTM/`
folder. The raw data (`harth/`, `har70plus/`, `Real world HAR/`) is expected
at the repo root, one level above `LSTM/`. `Real world HAR/` is not in git;
download it separately.

```bash
# from the repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, pyarrow, scipy, scikit-learn, matplotlib, torch
cd LSTM
python src/01_load_harth.py              # ~1 min
python src/02_load_har70.py
python src/03_load_realworld.py          # ~2 min
python src/04_labels.py
python src/05_windows.py                 # writes windows.npz (510 MB), folds.csv, scalers.json
python src/06_train_lstm.py              # ~35 min on Apple GPU; results in processed/results/
```

Every script prints its sanity checks (row counts, label counts, segment
statistics) and writes a plot to `processed/plots/`. All settings (window
length, overlap, folds, seed, model size, augmentation, which datasets to
train on) are in `src/config.py`.
