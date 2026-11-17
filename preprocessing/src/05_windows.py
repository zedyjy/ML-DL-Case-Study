# %% [markdown]
# # Step 5: sliding windows, subject folds and the scaler
#
# Input : `processed/all_labelled.parquet`
# Output: `processed/windows.npz`   - the model input
#         `processed/folds.csv`     - which subject is in which fold
#         `processed/scalers.json`  - per-channel mean / std per fold
#
# A neural network cannot digest a 2-hour recording at once. We cut each
# gap-free segment into short overlapping *windows* of 128 samples (2.56 s
# at 50 Hz), sliding forward 64 samples (50 % overlap) each time. Each
# window becomes one training example of shape (128, 6): 128 time steps x
# 6 channels (back x/y/z, thigh x/y/z). Its label is the majority label of
# the samples inside it.
#
# Why overlap? It doubles the number of examples and makes the model see
# every activity at many different phase positions (a step starting at the
# beginning, in the middle, at the end of the window ...).
#
# The split is by SUBJECT, never by window: neighbouring windows overlap by
# half, so putting windows of the same person in both train and test would
# be leakage and give a flattering score.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

W, STEP = config.WINDOW_SIZE, config.WINDOW_STEP
N_CLASSES = len(config.CLASSES)

# %% [markdown]
# ## Load

# %%
data = pd.read_parquet(config.PROCESSED_DIR / "all_labelled.parquet")
print(f"Loaded {len(data):,} rows")

# %% [markdown]
# ## Cut windows, segment by segment
#
# For a segment with n samples the windows start at 0, 64, 128, ... as long
# as start + 128 <= n. `np.lib.stride_tricks.sliding_window_view` gives us
# all windows of a 1-D index array without copying; we then take every
# 64th one. Labels: for each window we count how often each class occurs
# (one-hot sum) and pick the most frequent class. `purity` is that class's
# share of the window (1.0 = all 128 samples agree).

# %%
sensor_values = data[config.SENSOR_COLS].to_numpy(dtype=np.float32)
label_values = data["label_id"].to_numpy()

X_parts, y_parts, purity_parts, meta_parts = [], [], [], []
# `segment` ids are sorted, and rows of a segment are contiguous because
# steps 1-4 built the table that way; we still look them up explicitly.
seg_bounds = data.groupby("segment", sort=False).indices   # segment -> row indices
n_skipped = 0
for seg_id, rows in seg_bounds.items():
    n = len(rows)
    if n < W:
        n_skipped += 1
        continue
    starts = np.arange(0, n - W + 1, STEP)
    idx = rows[0] + starts[:, None] + np.arange(W)[None, :]      # (n_win, 128) row numbers

    X_parts.append(sensor_values[idx])                            # (n_win, 128, 6)
    counts = (label_values[idx][:, :, None] == np.arange(N_CLASSES)).sum(axis=1)  # (n_win, 6)
    y_parts.append(counts.argmax(axis=1).astype(np.int8))
    purity_parts.append((counts.max(axis=1) / W).astype(np.float32))

    first = data.iloc[rows[0]]
    meta_parts.append(pd.DataFrame({
        "dataset": first["dataset"], "subject": first["subject"], "segment": seg_id,
        "start_row": rows[0] + starts,
    }))

X = np.concatenate(X_parts)
y = np.concatenate(y_parts)
purity = np.concatenate(purity_parts)
meta = pd.concat(meta_parts, ignore_index=True)
print(f"Windows: {len(X):,}  (shape {X.shape}); segments too short and skipped: {n_skipped}")

# %% [markdown]
# ## Subject folds
#
# Training subjects (HARTH + RealWorld) are shuffled with a fixed seed and
# dealt round-robin into N_FOLDS groups, separately per dataset so every
# fold contains both datasets in the same proportion. HAR70+ subjects get
# fold -1: they are the zero-shot test set and never used for training,
# validation or the scaler.

# %%
rng = np.random.default_rng(config.SPLIT_SEED)
fold_of_subject = {}
for ds in ["harth", "realworld"]:
    subjects = sorted(meta.loc[meta["dataset"] == ds, "subject"].unique())
    rng.shuffle(subjects)
    for i, s in enumerate(subjects):
        fold_of_subject[s] = i % config.N_FOLDS
for s in meta.loc[meta["dataset"] == "har70plus", "subject"].unique():
    fold_of_subject[s] = config.TEST_FOLD

meta["fold"] = meta["subject"].map(fold_of_subject).astype(np.int8)
folds = (pd.Series(fold_of_subject, name="fold").rename_axis("subject").reset_index()
         .assign(dataset=lambda d: d["subject"].str.split("_").str[0]).sort_values(["fold", "subject"]))
folds.to_csv(config.PROCESSED_DIR / "folds.csv", index=False)

print("\nSubjects per fold:")
for f, g in folds.groupby("fold"):
    print(f"  fold {f:>2}: {', '.join(g['subject'])}")

# %% [markdown]
# ## Scaler: per-channel mean and std from training subjects only
#
# Standardising (subtract mean, divide by std) puts every channel on a
# similar scale, which helps the optimiser. The statistics must come from
# training data only, otherwise information about the test subjects leaks
# into the preprocessing. With K folds that means one scaler per fold
# (fitted on the K-1 training folds) plus one fitted on all training
# subjects for the final model. HAR70+ is excluded from all of them.
#
# Note for step 6: the rotation augmentation about x must be applied to the
# raw signal BEFORE scaling (scaling y and z differently would distort a
# rotation). Step 6 therefore receives unscaled X and applies the scaler
# itself.

# %%
def channel_stats(mask):
    sub = X[mask]                                   # (n, 128, 6)
    return {"mean": sub.mean(axis=(0, 1)).tolist(), "std": sub.std(axis=(0, 1)).tolist()}

is_train_pool = meta["fold"].to_numpy() != config.TEST_FOLD
scalers = {"all": channel_stats(is_train_pool)}
for f in range(config.N_FOLDS):
    scalers[f"fold_{f}"] = channel_stats(is_train_pool & (meta["fold"].to_numpy() != f))
with open(config.PROCESSED_DIR / "scalers.json", "w") as fh:
    json.dump({"channels": config.SENSOR_COLS, **scalers}, fh, indent=2)

print("\nScaler fitted on all training subjects (per channel):")
print(pd.DataFrame(scalers["all"], index=config.SENSOR_COLS).round(3).to_string())

# %% [markdown]
# ## Sanity checks

# %%
print("\nWindows per class and dataset:")
tab = pd.crosstab(meta["dataset"], pd.Series(y, name="class").map(dict(enumerate(config.CLASSES))))
print(tab.reindex(columns=config.CLASSES).to_string())

print("\nWindows per fold:")
print(meta["fold"].value_counts().sort_index().to_string())

print(f"\nWindow purity: {np.mean(purity == 1.0) * 100:.1f} % of windows have a single label, "
      f"{np.mean(purity < 0.5) * 100:.2f} % have a majority below 50 %")

assert not np.isnan(X).any(), "NaN in X"
assert X.shape[1:] == (W, 6)
assert set(np.unique(y)) <= set(range(N_CLASSES))
print("\nChecks passed: no NaN, shape (N, 128, 6), labels in 0..5")

# %%
# Plot: one window of each class from HARTH (top row) and HAR70+ (bottom
# row), thigh_x and back_x. A useful first look at how the classes differ
# and how the older adults' signal compares.
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, N_CLASSES, figsize=(18, 5), sharex=True, sharey=True)
t = np.arange(W) / config.SAMPLE_RATE_HZ
for r, ds in enumerate(["harth", "har70plus"]):
    for c, cls in enumerate(config.CLASSES):
        cand = np.where((meta["dataset"] == ds) & (y == c) & (purity == 1.0))[0]
        i = cand[len(cand) // 2]
        axes[r, c].plot(t, X[i, :, 3], label="thigh_x", lw=0.8)
        axes[r, c].plot(t, X[i, :, 0], label="back_x", lw=0.8, alpha=0.7)
        axes[r, c].set_title(f"{ds}: {cls}", fontsize=9)
axes[0, 0].legend(fontsize=7)
axes[1, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("g"); axes[1, 0].set_ylabel("g")
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "05_example_windows.png", dpi=110)
if "ipykernel" in sys.modules:
    plt.show()

# %% [markdown]
# ## Save
#
# `np.savez` stores several arrays in one file. Strings (subject, dataset,
# segment) are stored as numpy string arrays. Load with
# `d = np.load(path, allow_pickle=False); X = d["X"]; ...`

# %%
out_path = config.PROCESSED_DIR / "windows.npz"
np.savez(
    out_path,
    X=X,                                    # (N, 128, 6) float32, in g, NOT scaled
    y=y,                                    # (N,) int8, index into config.CLASSES
    purity=purity,                          # (N,) float32
    subject=meta["subject"].to_numpy().astype(str),
    dataset=meta["dataset"].to_numpy().astype(str),
    segment=meta["segment"].to_numpy().astype(str),
    fold=meta["fold"].to_numpy(),           # (N,) int8, -1 = HAR70+ test
    classes=np.array(config.CLASSES),
)
print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
