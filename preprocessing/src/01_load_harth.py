# %% [markdown]
# # Step 1: load HARTH into one clean table
#
# Input : `harth/S006.csv` ... `harth/S029.csv` (22 subjects, raw, read-only)
# Output: `processed/harth.parquet`
#
# What this script does, in order:
#  1. Read every CSV and fix the three header variants (some files carry an
#     extra index column that we do not need).
#  2. Turn the timestamp text into "seconds since the file started".
#  3. Fix S006, whose timestamps step 10 ms although the data is 50 Hz
#     (see the check in the "S006" cell below).
#  4. Split each subject's recording into gap-free *segments*. A gap is a
#     jump in time larger than `GAP_THRESHOLD_S`. Later, windows are cut
#     inside a segment only, so a window never contains a time jump.
#  5. Give every row a clean time axis `t` (seconds since the segment began,
#     exactly 1/50 s apart) and save everything as one Parquet file.
#
# Labels are kept as the original HARTH codes (1 = walking, 7 = sitting ...).
# The mapping to a common label set happens in step 4.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# When this file runs as a plain script (python src/01_load_harth.py) there is
# no notebook kernel, so plt.show() would open a window and block. Switch to
# the "Agg" backend in that case: figures are only saved to disk. In VS Code's
# interactive window (cell by cell) the kernel is loaded and plots show inline.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `import config` work no matter where the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import split_into_segments, print_segment_stats, first_long_run

# %% [markdown]
# ## Settings for this step

# %%
# The 8 columns every HARTH file should end up with, in this order.
EXPECTED_COLS = ["timestamp"] + config.SENSOR_COLS + ["label"]

# S006 stores timestamps 10 ms apart, but the signal itself is 50 Hz (its
# walking cadence is only normal if we read it at 50 Hz; at 100 Hz it would be
# 3.8 steps per second, which is sprinting). Multiplying its elapsed time by 2
# puts it on the same 20 ms spacing as every other subject. Set the factor to
# 1 to switch this correction off.
TIMESTAMP_FACTOR = {"S006": 2.0}

# %% [markdown]
# ## Helper 1: read one file and fix the header
#
# Three header variants exist:
#  - normal: `timestamp, back_x, ..., label`
#  - S015, S021: an extra `index` column after `timestamp`
#  - S023: an unnamed first column (pandas calls it `Unnamed: 0`)
#
# The fix is the same for all: keep only the 8 columns we expect.

# %%
def read_harth_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    extra = [c for c in df.columns if c not in EXPECTED_COLS]
    if extra:
        print(f"  {path.name}: dropping extra column(s) {extra}")
    df = df[EXPECTED_COLS]

    # The timestamps are text like "2019-01-12 00:00:00.020". pandas parses
    # them into real datetime objects (nanosecond precision, so S020's
    # 9-digit fractions are fine). format="ISO8601" tells pandas the layout
    # up front, which is much faster than letting it guess 400k times.
    ts = pd.to_datetime(df["timestamp"], format="ISO8601")

    # We never need the absolute date (it is fake in most files anyway:
    # 2019-01-12 or 2000-01-01). What matters is elapsed time in seconds.
    elapsed = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()

    subject = path.stem  # "S006"
    elapsed = elapsed * TIMESTAMP_FACTOR.get(subject, 1.0)

    out = pd.DataFrame({
        "subject": "harth_" + subject,
        "elapsed": elapsed,
        "label": df["label"].astype("int16"),
    })
    for c in config.SENSOR_COLS:
        out[c] = df[c].astype("float32")
    return out


# %% [markdown]
# ## Helper 2: split one subject into gap-free segments
#
# `split_into_segments` lives in `common.py` because step 2 (HAR70+) needs
# exactly the same logic. In short: wherever two consecutive rows are more
# than `GAP_THRESHOLD_S` apart, a new segment starts, and each segment gets
# a clean time axis `t` = 0, 0.02, 0.04 ...

# %% [markdown]
# ## Run it for all 22 files

# %%
files = sorted(config.HARTH_DIR.glob("S*.csv"))
print(f"Found {len(files)} HARTH files")

parts = []
for f in files:
    df = read_harth_file(f)
    df = split_into_segments(df)
    n_seg = df["segment"].nunique()
    print(f"  {f.name}: {len(df):>8,} rows, {df['elapsed'].iloc[-1] / 60:6.1f} min, "
          f"{n_seg:>3} segments")
    parts.append(df)

harth = pd.concat(parts, ignore_index=True)
harth.insert(0, "dataset", "harth")

# Final column order = the target format in CLAUDE.md section 4.
harth = harth[["dataset", "subject", "segment", "t"] + config.SENSOR_COLS + ["label"]]

# %% [markdown]
# ## Sanity checks
#
# Numbers to compare against CLAUDE.md section 3.1:
# 6,461,328 rows and the per-label row counts listed there.

# %%
print("\nShape:", harth.shape)
print("Row count matches CLAUDE.md (6,461,328):", len(harth) == 6_461_328)
print("\nDtypes:\n", harth.dtypes.to_string())

print("\nLabel counts (original HARTH codes):")
counts = harth["label"].value_counts().sort_index()
for code, n in counts.items():
    print(f"  {code:>4} {config.HARTH_LABEL_NAMES[code]:<24} {n:>10,}")

# %%
# Segment statistics: how many segments, how long are they?
print_segment_stats(harth)

# The sampling-rate check already ran inside split_into_segments: for every
# subject, the raw timestamps inside a segment deviate < 5 ms from a perfect
# 50 Hz grid (the assert would have stopped the script otherwise).
print("\nSampling rate check: passed for all subjects (see per-file output above)")

# %%
# A short signal plot: 10 seconds of walking from S008 (normal timestamps)
# and from S006 (corrected timestamps). If the S006 fix is right, both show
# roughly 2 steps per second.
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, subj in zip(axes, ["harth_S008", "harth_S006"]):
    # first unbroken run of walking (label 1) that is at least 10 s long
    g = first_long_run(harth[harth["subject"] == subj], label=1, min_rows=500).iloc[:500]
    ax.plot(g["t"] - g["t"].iloc[0], g["thigh_x"], label="thigh_x")
    ax.plot(g["t"] - g["t"].iloc[0], g["back_x"], label="back_x", alpha=0.7)
    ax.set_title(f"{subj}: 10 s of walking (label 1), segment {g['segment'].iloc[0]}")
    ax.set_ylabel("acceleration [g]")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "01_harth_walking.png", dpi=120)
if "ipykernel" in sys.modules:
    plt.show()  # inline in VS Code; as a script the PNG above is enough

# %% [markdown]
# ## Save

# %%
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.PROCESSED_DIR / "harth.parquet"
harth.to_parquet(out_path, index=False)
print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
print(harth.head())
