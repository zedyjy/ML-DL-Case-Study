# %% [markdown]
# # config.py: one place for every setting the pipeline shares
#
# Every script in `src/` imports from this file, so if we ever change the
# window length, the sampling rate or a folder name, we change it here once
# and all scripts pick it up.

# %%
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# `Path(__file__)` is the path of this config.py file. Its parent is `src/`,
# and the parent of that is the `LSTM/` folder. Building paths this way means
# no folder name on anyone's computer ever has to appear in the code.
#
# Repo layout (the raw data is shared by every model, so it sits at the
# repo root, one level above `LSTM/`):
#   ML-DL-Case-Study/
#   ├── harth/  har70plus/  Real world HAR/   <- DATA_DIR
#   └── LSTM/                                 <- LSTM_DIR
#       ├── src/        (this file)
#       └── processed/  (generated, not in git)
LSTM_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LSTM_DIR.parent

# Raw data: read-only. Never write into these folders.
HARTH_DIR = DATA_DIR / "harth"
HAR70_DIR = DATA_DIR / "har70plus"
REALWORLD_DIR = DATA_DIR / "Real world HAR"

# Everything we generate goes here (Parquet tables, .npz window files,
# scalers, trained models, plots). Kept inside LSTM/ so it never mixes
# with other team members' outputs.
PROCESSED_DIR = LSTM_DIR / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"

# ---------------------------------------------------------------------------
# Signal settings
# ---------------------------------------------------------------------------
# All three datasets are (nominally) recorded at 50 samples per second.
SAMPLE_RATE_HZ = 50

# Window length in samples. 128 samples / 50 Hz = 2.56 seconds. This is a
# common choice in HAR papers (e.g. the UCI-HAR dataset uses exactly this).
WINDOW_SIZE = 128

# How far the window slides forward each time. 50 % overlap means the next
# window starts in the middle of the previous one, so step = 64 samples.
WINDOW_OVERLAP = 0.5
WINDOW_STEP = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# A "gap" is a jump between two consecutive timestamps that is clearly larger
# than one sample period (20 ms at 50 Hz). Anything above this threshold
# splits the recording into two separate segments, so that no window ever
# contains a time jump. 0.1 s = 5 missing samples; HAR70+ jitters only
# between 19 and 21 ms, so this threshold leaves normal jitter alone.
GAP_THRESHOLD_S = 0.1

# ---------------------------------------------------------------------------
# Subject split (decision 4, agreed 17 Nov 2026)
# ---------------------------------------------------------------------------
# Subject-grouped K-fold cross-validation over the training subjects
# (HARTH + RealWorld). Every subject is held out in exactly one fold, so a
# model is never scored on a person it has seen. Set N_FOLDS = 37 for full
# leave-one-subject-out. HAR70+ is never part of any fold (fold = -1).
N_FOLDS = 5
SPLIT_SEED = 42      # fixed so the folds are identical for every team member
TEST_FOLD = -1       # fold value that marks HAR70+ windows

# ---------------------------------------------------------------------------
# LSTM training (step 6)
# ---------------------------------------------------------------------------
# Which datasets the model learns from. ["harth"] alone answers decision 5
# (does RealWorld help?); HAR70+ is never allowed here.
TRAIN_DATASETS = ["harth", "realworld"]

# Share of the *training subjects* of a fold that are set aside for early
# stopping (chosen by subject, not by window). The held-out fold itself is
# never used for any decision during training.
VAL_SUBJECT_FRACTION = 0.15

# Model size. hidden 64 x 2 layers is small (~55k weights): fast on a laptop
# and hard to overfit with 120k windows.
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3

# Optimisation
BATCH_SIZE = 256
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 5      # stop when val macro-F1 has not improved for 5 epochs

# Augmentation: rotate each window by a random angle about the x (vertical)
# axis, separately for back and thigh (decision 2). Makes the model ignore
# how the sensor was turned around the body axis.
AUGMENT_ROTATION = True

# Random seed for weights init, shuffling and the validation subject pick.
TRAIN_SEED = 0

MODELS_DIR = PROCESSED_DIR / "models"
RESULTS_DIR = PROCESSED_DIR / "results"

# ---------------------------------------------------------------------------
# Column names of the harmonized tables (section 4 of CLAUDE.md)
# ---------------------------------------------------------------------------
SENSOR_COLS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

# Gravity in m/s^2; RealWorld stores m/s^2 and must be divided by this to get g.
G_MS2 = 9.80665

# ---------------------------------------------------------------------------
# Original label codes (for reference and for the mapping in step 4)
# ---------------------------------------------------------------------------
# HARTH and HAR70+ share the same code book.
HARTH_LABEL_NAMES = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling_sit",
    14: "cycling_stand",
    130: "cycling_sit_inactive",
    140: "cycling_stand_inactive",
}

# RealWorld labels are the activity names in the file names.
REALWORLD_ACTIVITIES = [
    "walking", "running", "sitting", "standing", "lying",
    "climbingup", "climbingdown", "jumping",
]

# ---------------------------------------------------------------------------
# Common label set (decision 1, agreed 16 Nov 2026)
# ---------------------------------------------------------------------------
# We keep only the six activities that all three datasets share. Everything
# else maps to None and is dropped in step 4:
#   running   - HARTH and RealWorld only, absent in HAR70+ (the test set)
#   shuffling - HARTH and HAR70+ only, absent in RealWorld; also the class
#               the original HARTH models confused most
#   cycling   - HARTH only
#   jumping   - RealWorld only
# The ORDER of this list is the class index the model will predict
# (0 = walking, 1 = stairs_up, ...). Do not reorder it after training.
CLASSES = ["walking", "stairs_up", "stairs_down", "standing", "sitting", "lying"]

# HARTH and HAR70+: original numeric code -> common name
LABEL_MAP_HARTH = {
    1: "walking",
    2: None,            # running
    3: None,            # shuffling
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: None, 14: None, 130: None, 140: None,   # cycling variants
}
LABEL_MAP_HAR70 = LABEL_MAP_HARTH   # same code book

# RealWorld: activity name from the file name -> common name
LABEL_MAP_REALWORLD = {
    "walking": "walking",
    "climbingup": "stairs_up",
    "climbingdown": "stairs_down",
    "standing": "standing",
    "sitting": "sitting",
    "lying": "lying",
    "running": None,
    "jumping": None,
}
