# %% [markdown]
# # lstm_config.py: settings for the LSTM only (step 6)
#
# Data, window and split settings shared by every model live in
# `preprocessing/src/config.py`. Only what is specific to this model is here,
# so changing the LSTM never changes the data other team members use.

# %%
from pathlib import Path

# This folder (`LSTM/`). Trained weights and result tables are written
# inside it, not into the shared `preprocessing/processed/`.
LSTM_DIR = Path(__file__).resolve().parent

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

MODELS_DIR = LSTM_DIR / "models"
RESULTS_DIR = LSTM_DIR / "results"

