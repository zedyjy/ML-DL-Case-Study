# %% [markdown]
# # Step 6: LSTM in PyTorch - train on younger adults, test on HAR70+
#
# Input : `processed/windows.npz`, `processed/scalers.json`
# Output: `processed/results/lstm_*.json|csv|png`, `processed/models/lstm_*.pt`
#
# What happens here, in order:
#  1. Load the windows. Keep only the training datasets (HARTH + RealWorld)
#     for learning; HAR70+ is loaded too but only ever *scored*.
#  2. For each of the 5 folds: train on 4 folds, stop early using a small
#     validation slice of those training subjects, then score the held-out
#     fold (younger adults the model never saw) AND HAR70+ (older adults).
#  3. Train a final model on all training subjects and score HAR70+.
#  4. Report macro-F1, per-class recall and confusion matrices, plus the
#     "generalization gap" = held-out younger adults minus HAR70+.
#
# ------------------------------------------------------------------------
# A 2-minute LSTM refresher, in the words of this script
# ------------------------------------------------------------------------
# One window is a sequence of 128 time steps, each a vector of 6 numbers.
# An LSTM reads the sequence one step at a time and carries a *hidden
# state* (here 64 numbers) from step to step. At every step it decides,
# using small learned "gates", what to forget from the state, what to add
# from the current input, and what to output. After the last step the
# hidden state is a 64-number summary of the whole 2.56 s. A final linear
# layer turns that summary into 6 scores, one per class; the largest score
# is the prediction. Training = nudging all the weights so that the score
# of the correct class goes up (cross-entropy loss + Adam optimiser).

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from evaluate import score, print_report, plot_confusion

# Apple-silicon GPU ("mps") if available, otherwise CPU.
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training on", DEVICE)

# Reproducibility: same seed -> same weight init, same shuffling.
torch.manual_seed(config.TRAIN_SEED)
np.random.seed(config.TRAIN_SEED)

N_CLASSES = len(config.CLASSES)
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load windows

# %%
d = np.load(config.PROCESSED_DIR / "windows.npz", allow_pickle=False)
X_all, y_all = d["X"], d["y"].astype(np.int64)
subject_all, dataset_all, fold_all = d["subject"], d["dataset"], d["fold"]
with open(config.PROCESSED_DIR / "scalers.json") as fh:
    scalers = json.load(fh)

is_test = dataset_all == "har70plus"
is_pool = np.isin(dataset_all, config.TRAIN_DATASETS)   # everything we may learn from
print(f"Windows: {is_pool.sum():,} training pool ({config.TRAIN_DATASETS}), "
      f"{is_test.sum():,} HAR70+ test")


# %% [markdown]
# ## 2. The model

# %%
class LSTMClassifier(nn.Module):
    def __init__(self, n_channels=6, hidden=config.LSTM_HIDDEN,
                 layers=config.LSTM_LAYERS, dropout=config.LSTM_DROPOUT, n_classes=N_CLASSES):
        super().__init__()
        # batch_first=True means the input is (batch, time, channels), which
        # is exactly the (N, 128, 6) layout of our windows.
        # dropout between the stacked LSTM layers fights overfitting.
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):                 # x: (batch, 128, 6)
        out, _ = self.lstm(x)             # out: (batch, 128, hidden) - the state at every step
        last = out[:, -1, :]              # the state after the final time step: (batch, hidden)
        return self.head(self.dropout(last))   # (batch, 6) class scores ("logits")


# %% [markdown]
# ## 3. Preprocessing on the fly: rotation augmentation, then scaling
#
# Both run on the GPU on each batch, so no copy of the 500 MB array is made.
#
# Rotation about x: the x axis is "down" after step 3, so turning the
# sensor around that axis mixes y and z but leaves x alone:
#     y' = y cos(a) - z sin(a),   z' = y sin(a) + z cos(a)
# A different random angle for back and thigh, per window. Only during
# training - the model is scored on the data as it is.

# %%
MEAN = {k: torch.tensor(v["mean"], device=DEVICE) for k, v in scalers.items() if k != "channels"}
STD = {k: torch.tensor(v["std"], device=DEVICE) for k, v in scalers.items() if k != "channels"}


def rotate_about_x(x):
    """x: (batch, 128, 6) raw g. Random rotation of (y, z) per sensor per window."""
    batch = x.shape[0]
    out = x.clone()
    for sensor_start in (0, 3):                                     # back = cols 0..2, thigh = 3..5
        angle = torch.rand(batch, 1, device=x.device) * 2 * np.pi   # 0 .. 360 degrees
        cos, sin = torch.cos(angle), torch.sin(angle)
        y = x[:, :, sensor_start + 1]
        z = x[:, :, sensor_start + 2]
        out[:, :, sensor_start + 1] = y * cos - z * sin
        out[:, :, sensor_start + 2] = y * sin + z * cos
    return out


def scale(x, scaler_name):
    return (x - MEAN[scaler_name]) / STD[scaler_name]


# %% [markdown]
# ## 4. One training run
#
# `run_training` is used 6 times (5 folds + final). It returns the trained
# model with the weights of the epoch that had the best validation macro-F1
# (that is what "early stopping" means: we keep training a few more epochs
# in case it improves again, but restore the best weights at the end).

# %%
def make_loader(idx, shuffle):
    """Batches of (window, label) for the given row indices."""
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_all[idx]), torch.from_numpy(y_all[idx]))
    return torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def predict(model, idx, scaler_name):
    """Class index for every window in idx (no gradient tracking needed)."""
    model.eval()
    preds = []
    for xb, _ in make_loader(idx, shuffle=False):
        logits = model(scale(xb.to(DEVICE), scaler_name))
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def class_weights(y):
    """Inverse-frequency weights so rare classes (stairs) matter in the loss.

    A class with 1 % of the windows gets about 50x the weight of one with
    50 %. Normalised so the weights average to 1.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = counts.sum() / np.maximum(counts, 1) / N_CLASSES
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)


def run_training(train_idx, val_idx, scaler_name, tag):
    model = LSTMClassifier().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_all[train_idx]))
    train_loader = make_loader(train_idx, shuffle=True)

    best_f1, best_state, epochs_without_gain = -1.0, None, 0
    history = []
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if config.AUGMENT_ROTATION:
                xb = rotate_about_x(xb)
            xb = scale(xb, scaler_name)

            optimiser.zero_grad()            # forget gradients of the previous batch
            loss = loss_fn(model(xb), yb)    # how wrong are we on this batch?
            loss.backward()                  # gradients of the loss w.r.t. every weight
            optimiser.step()                 # move the weights a little against the gradient
            total_loss += loss.item() * len(xb)

        val_f1 = score(y_all[val_idx], predict(model, val_idx, scaler_name))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_idx), "val_macro_f1": val_f1})
        print(f"  [{tag}] epoch {epoch:2d}  loss {total_loss / len(train_idx):.4f}  "
              f"val macro-F1 {val_f1:.3f}  ({time.time() - t0:.0f} s)")

        if val_f1 > best_f1:
            best_f1, epochs_without_gain = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.EARLY_STOP_PATIENCE:
                print(f"  [{tag}] early stop: no improvement for {config.EARLY_STOP_PATIENCE} epochs")
                break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(config.RESULTS_DIR / f"lstm_{tag}_history.csv", index=False)
    return model


def split_off_validation(pool_idx, rng):
    """Pick VAL_SUBJECT_FRACTION of the *subjects* in pool_idx for validation."""
    subjects = np.unique(subject_all[pool_idx])
    n_val = max(2, int(round(len(subjects) * config.VAL_SUBJECT_FRACTION)))
    val_subjects = rng.choice(subjects, size=n_val, replace=False)
    is_val = np.isin(subject_all[pool_idx], val_subjects)
    return pool_idx[~is_val], pool_idx[is_val], sorted(map(str, val_subjects))


# %% [markdown]
# ## 5. Five-fold cross-validation
#
# For fold k: training pool = training datasets minus fold k. A few of
# those subjects become the validation set for early stopping. The model
# is then scored on fold k (younger adults it never saw) and on HAR70+.

# %%
rng = np.random.default_rng(config.TRAIN_SEED)
test_idx = np.where(is_test)[0]
fold_results = []
for k in range(config.N_FOLDS):
    heldout_idx = np.where(is_pool & (fold_all == k))[0]
    pool_idx = np.where(is_pool & (fold_all != k) & (fold_all != config.TEST_FOLD))[0]
    train_idx, val_idx, val_subjects = split_off_validation(pool_idx, rng)
    print(f"\n--- fold {k}: train {len(train_idx):,} windows, val {len(val_idx):,} "
          f"(subjects {val_subjects}), held-out {len(heldout_idx):,} ---")

    model = run_training(train_idx, val_idx, f"fold_{k}", tag=f"fold{k}")
    res_heldout = score(y_all[heldout_idx], predict(model, heldout_idx, f"fold_{k}"))
    res_har70 = score(y_all[test_idx], predict(model, test_idx, f"fold_{k}"))
    print_report(res_heldout, f"fold {k}: held-out younger adults")
    print_report(res_har70, f"fold {k}: HAR70+ (zero-shot)")
    fold_results.append({"fold": k, "heldout": res_heldout, "har70plus": res_har70})
    torch.save(model.state_dict(), config.MODELS_DIR / f"lstm_fold{k}.pt")

# %%
summary = pd.DataFrame({
    "fold": [r["fold"] for r in fold_results],
    "heldout_macro_f1": [r["heldout"]["macro_f1"] for r in fold_results],
    "har70_macro_f1": [r["har70plus"]["macro_f1"] for r in fold_results],
})
summary["gap"] = summary["heldout_macro_f1"] - summary["har70_macro_f1"]
print("\n=== Cross-validation summary (macro-F1) ===")
print(summary.round(3).to_string(index=False))
print(f"\nheld-out younger adults: {summary.heldout_macro_f1.mean():.3f} +- {summary.heldout_macro_f1.std():.3f}")
print(f"HAR70+ (older adults):   {summary.har70_macro_f1.mean():.3f} +- {summary.har70_macro_f1.std():.3f}")
print(f"generalization gap:      {summary.gap.mean():.3f}")

# Per-class recall averaged over folds: which activities break on HAR70+?
recall = pd.DataFrame({
    "heldout": pd.DataFrame([r["heldout"]["per_class_recall"] for r in fold_results]).mean(),
    "har70plus": pd.DataFrame([r["har70plus"]["per_class_recall"] for r in fold_results]).mean(),
})
print("\nPer-class recall, mean over folds:")
print(recall.round(3).to_string())

# %% [markdown]
# ## 6. Final model on all training subjects -> HAR70+

# %%
pool_idx = np.where(is_pool)[0]
train_idx, val_idx, val_subjects = split_off_validation(pool_idx, rng)
print(f"\n--- final: train {len(train_idx):,} windows, val {len(val_idx):,} (subjects {val_subjects}) ---")
final_model = run_training(train_idx, val_idx, "all", tag="final")
res_final = score(y_all[test_idx], predict(final_model, test_idx, "all"))
print_report(res_final, "FINAL model: HAR70+ (zero-shot)")
torch.save(final_model.state_dict(), config.MODELS_DIR / "lstm_final.pt")

# %% [markdown]
# ## 7. Save everything

# %%
run_name = "lstm_" + "+".join(config.TRAIN_DATASETS) + ("_rot" if config.AUGMENT_ROTATION else "_norot")
summary.to_csv(config.RESULTS_DIR / f"{run_name}_cv.csv", index=False)
with open(config.RESULTS_DIR / f"{run_name}.json", "w") as fh:
    json.dump({"config": {k: getattr(config, k) for k in
                          ["TRAIN_DATASETS", "WINDOW_SIZE", "N_FOLDS", "LSTM_HIDDEN", "LSTM_LAYERS",
                           "LSTM_DROPOUT", "BATCH_SIZE", "MAX_EPOCHS", "LEARNING_RATE", "AUGMENT_ROTATION"]},
               "folds": fold_results, "final_har70plus": res_final}, fh, indent=2)
plot_confusion(res_final, f"{run_name}: HAR70+ (final model)", config.RESULTS_DIR / f"{run_name}_cm_har70.png")
# pooled confusion over the 5 held-out folds (every training subject once)
pooled = {"confusion": np.sum([r["heldout"]["confusion"] for r in fold_results], axis=0).tolist()}
plot_confusion(pooled, f"{run_name}: held-out younger adults (5 folds)", config.RESULTS_DIR / f"{run_name}_cm_heldout.png")
print(f"\nSaved results to {config.RESULTS_DIR}/{run_name}*")
