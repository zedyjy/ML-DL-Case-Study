# %% [markdown]
# # Step 4: map every dataset to the common label set
#
# Input : `processed/harth.parquet`, `processed/har70plus.parquet`,
#         `processed/realworld.parquet` (labels still in their original form:
#         numeric codes for HARTH / HAR70+, activity names for RealWorld)
# Output: `processed/all_labelled.parquet` - one table, all datasets, with
#         `label` = common class name and `label_id` = class index 0..5
#
# The mapping tables live in `config.py` (LABEL_MAP_*) so that changing the
# class set is a one-file edit. Rows whose activity maps to None (running,
# shuffling, cycling, jumping) are dropped here.
#
# One subtlety: dropping rows can create new "holes" inside a segment. For
# example a HARTH segment might go walking -> running -> walking; after we
# drop running, the two walking parts are no longer continuous in time and
# a window must not span the hole. So after dropping we re-split segments
# wherever the time axis `t` jumps.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# %% [markdown]
# ## Load the three tables and apply the mapping

# %%
tables = {
    "harth": (config.PROCESSED_DIR / "harth.parquet", config.LABEL_MAP_HARTH),
    "har70plus": (config.PROCESSED_DIR / "har70plus.parquet", config.LABEL_MAP_HAR70),
    "realworld": (config.PROCESSED_DIR / "realworld.parquet", config.LABEL_MAP_REALWORLD),
}

parts = []
for name, (path, label_map) in tables.items():
    df = pd.read_parquet(path)
    n_before = len(df)

    # Every original label must be in the mapping table; an unknown label
    # would silently become NaN otherwise, so we check explicitly.
    unknown = set(df["label"].unique()) - set(label_map)
    assert not unknown, f"{name}: labels {unknown} are not in the mapping table"

    df["label_orig"] = df["label"]
    df["label"] = df["label"].map(label_map)           # None -> NaN
    dropped = df["label"].isna()
    print(f"{name}: {n_before:,} rows, dropping {dropped.sum():,} "
          f"({dropped.mean() * 100:.1f} %) with unmapped labels")
    print("   dropped per original label:",
          df.loc[dropped, "label_orig"].value_counts().to_dict())
    df = df[~dropped].copy()
    parts.append(df)

data = pd.concat(parts, ignore_index=True)

# %% [markdown]
# ## Re-split segments at the holes left by dropped rows
#
# Inside a segment `t` advances by exactly 0.02 s. After dropping rows, a
# jump larger than that marks a hole. We append a running counter to the
# segment id so that both sides of the hole become separate segments.

# %%
dt = data.groupby("segment")["t"].diff()
hole = dt > (1.0 / config.SAMPLE_RATE_HZ) * 1.5      # anything above 1.5 samples
n_holes = int(hole.sum())
sub_id = hole.groupby(data["segment"]).cumsum().astype(int)
data["segment"] = data["segment"] + "_" + sub_id.astype(str).str.zfill(2)
# restart the time axis in each new segment
data["t"] = (data.groupby("segment").cumcount() / config.SAMPLE_RATE_HZ).astype("float32")
print(f"\nHoles created by dropping rows: {n_holes} -> segments now: {data['segment'].nunique()}")

# %% [markdown]
# ## Class index
#
# Neural networks predict a number, not a word. `label_id` is the position
# of the class name in `config.CLASSES`, so 0 = walking, ..., 5 = lying.

# %%
data["label_id"] = data["label"].map({c: i for i, c in enumerate(config.CLASSES)}).astype("int8")
data["label"] = data["label"].astype("category")
data = data[["dataset", "subject", "segment", "t"] + config.SENSOR_COLS + ["label", "label_id"]]

# %% [markdown]
# ## Sanity checks

# %%
print("\nShape:", data.shape)
print("Subjects per dataset:", data.groupby("dataset", observed=True)["subject"].nunique().to_dict())

print("\nMinutes per class and dataset:")
minutes = (data.groupby(["label", "dataset"], observed=True).size() / 50 / 60).unstack("dataset")
minutes = minutes.reindex(config.CLASSES).round(1)
minutes["total"] = minutes.sum(axis=1)
print(minutes.to_string())

print("\nShare of each class within a dataset (%):")
share = data.groupby(["dataset", "label"], observed=True).size().unstack("label")
share = (share.div(share.sum(axis=1), axis=0) * 100).reindex(columns=config.CLASSES).round(1)
print(share.to_string())

# Classes a model trained on HARTH + RealWorld will see, vs. what HAR70+ asks
train_classes = set(data.loc[data["dataset"] != "har70plus", "label"].unique())
test_classes = set(data.loc[data["dataset"] == "har70plus", "label"].unique())
print("\nClasses in training data == classes in HAR70+:", train_classes == test_classes == set(config.CLASSES))

seg_len = data.groupby("segment").size()
print(f"\nSegments shorter than one window ({config.WINDOW_SIZE} samples): "
      f"{(seg_len < config.WINDOW_SIZE).sum()} of {len(seg_len)} "
      f"({seg_len[seg_len < config.WINDOW_SIZE].sum() / 50:.0f} s of data, will be skipped in step 5)")

# %% [markdown]
# ## Save

# %%
out_path = config.PROCESSED_DIR / "all_labelled.parquet"
data.to_parquet(out_path, index=False)
print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
print(data.head())
