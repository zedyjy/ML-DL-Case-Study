# %% [markdown]
# # Step 2: load HAR70+ into one clean table
#
# Input : `har70plus/501.csv` ... `har70plus/518.csv` (18 older adults, raw)
# Output: `processed/har70plus.parquet`
#
# HAR70+ uses the same sensors, axes, units (g) and label codes as HARTH, so
# this step is a simpler copy of step 1:
#  1. Read every CSV (all files have the clean 8-column header).
#  2. Turn the timestamp into seconds since the file started.
#  3. Split into gap-free segments and build the clean 50 Hz time axis `t`.
#  4. Save as Parquet with the original label codes.
#
# One difference: HAR70+ timestamps jitter between 19, 20 and 21 ms. Our
# survey showed the 19s and 21s come in equal numbers, so the average spacing
# is exactly 20 ms and the jitter never accumulates beyond 1 ms. That means
# we can treat the data as a clean 50 Hz grid; `split_into_segments` checks
# this (it stops if the drift inside any segment exceeds 5 ms).
#
# REMINDER: HAR70+ is the zero-shot TEST set. Nothing from it may be used for
# training, validation or for fitting the scaler.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # plain script: only save figures, never block
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import split_into_segments, print_segment_stats, first_long_run

# %% [markdown]
# ## Helper: read one file

# %%
EXPECTED_COLS = ["timestamp"] + config.SENSOR_COLS + ["label"]


def read_har70_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert list(df.columns) == EXPECTED_COLS, f"{path.name}: unexpected columns {list(df.columns)}"

    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    elapsed = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()

    out = pd.DataFrame({
        "subject": "har70_" + path.stem,  # "har70_501"
        "elapsed": elapsed,
        "label": df["label"].astype("int16"),
    })
    for c in config.SENSOR_COLS:
        out[c] = df[c].astype("float32")
    return out


# %% [markdown]
# ## Run it for all 18 files

# %%
files = sorted(config.HAR70_DIR.glob("5*.csv"))
print(f"Found {len(files)} HAR70+ files")

parts = []
for f in files:
    df = read_har70_file(f)
    df = split_into_segments(df)
    print(f"  {f.name}: {len(df):>8,} rows, {df['elapsed'].iloc[-1] / 60:6.1f} min, "
          f"{df['segment'].nunique():>3} segments")
    parts.append(df)

har70 = pd.concat(parts, ignore_index=True)
har70.insert(0, "dataset", "har70plus")
har70 = har70[["dataset", "subject", "segment", "t"] + config.SENSOR_COLS + ["label"]]

# %% [markdown]
# ## Sanity checks
#
# Numbers to compare against CLAUDE.md section 3.2:
# 2,259,597 rows; labels 1, 3, 4, 5, 6, 7, 8 only (no running, no cycling).

# %%
print("\nShape:", har70.shape)
print("Row count matches CLAUDE.md (2,259,597):", len(har70) == 2_259_597)

print("\nLabel counts (original codes):")
counts = har70["label"].value_counts().sort_index()
for code, n in counts.items():
    print(f"  {code:>4} {config.HARTH_LABEL_NAMES[code]:<12} {n:>10,}  ({n / 50 / 60:6.1f} min)")

print_segment_stats(har70)

# %%
# Orientation check: while standing (label 6), gravity should sit on the
# -x axis of both sensors (about -1 g), exactly like HARTH. If a subject
# wore a sensor upside down this is where we would see it.
print("\nMean acceleration while standing, per subject (expect back_x and thigh_x near -1):")
standing = har70[har70["label"] == 6].groupby("subject")[config.SENSOR_COLS].mean()
print(standing.round(2).to_string())

# %%
# Signal plot: 10 s of walking from one older adult next to 10 s of walking
# from a HARTH subject. Same axes, same units: the only thing that should
# differ is the person (older adults usually walk a bit slower / softer).
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
harth = pd.read_parquet(config.PROCESSED_DIR / "harth.parquet",
                        filters=[("subject", "==", "harth_S008")])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
for ax, (name, df) in zip(axes, [("harth_S008", harth), ("har70_501", har70)]):
    g = first_long_run(df[df["subject"] == name], label=1, min_rows=500).iloc[:500]
    ax.plot(g["t"] - g["t"].iloc[0], g["thigh_x"], label="thigh_x")
    ax.plot(g["t"] - g["t"].iloc[0], g["back_x"], label="back_x", alpha=0.7)
    ax.set_title(f"{name}: 10 s of walking, segment {g['segment'].iloc[0]}")
    ax.set_ylabel("acceleration [g]")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "02_har70_vs_harth_walking.png", dpi=120)
if "ipykernel" in sys.modules:
    plt.show()

# %% [markdown]
# ## Save

# %%
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.PROCESSED_DIR / "har70plus.parquet"
har70.to_parquet(out_path, index=False)
print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
print(har70.head())
