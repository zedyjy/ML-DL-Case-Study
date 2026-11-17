# %% [markdown]
# # common.py: helpers shared by several pipeline steps
#
# Only code that is used by more than one step lives here, so that each
# `0X_*.py` script stays readable on its own.

# %%
import numpy as np
import pandas as pd

import config


def split_into_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Split one subject's recording into gap-free segments.

    Expects columns `subject` and `elapsed` (seconds since the file started,
    one row per sample, nominally 1/50 s apart).

    Adds two columns:
      segment : "<subject>_000", "<subject>_001", ... one id per gap-free piece
      t       : seconds since the segment began, on a perfect 50 Hz grid

    `np.diff(elapsed)` is the time between consecutive rows. Normally it is
    0.02 s. Wherever it is larger than `GAP_THRESHOLD_S`, a new segment
    starts. `np.cumsum` of those "new segment here" flags gives a running
    counter that we use as the segment number.
    """
    dt = np.diff(df["elapsed"].to_numpy())
    is_gap = dt > config.GAP_THRESHOLD_S

    # First row always starts segment 0; every gap starts a new one.
    seg_num = np.concatenate([[0], np.cumsum(is_gap)])
    df = df.copy()
    df["segment"] = [f"{s}_{n:03d}" for s, n in zip(df["subject"], seg_num)]

    # Clean time axis per segment: 0, 0.02, 0.04, ... Inside a segment the
    # real spacing is 20 ms on average (we verify that below), so this only
    # removes millisecond jitter and makes every segment start at t = 0.
    row_in_seg = df.groupby("segment").cumcount()
    df["t"] = (row_in_seg / config.SAMPLE_RATE_HZ).astype("float32")

    # Sanity check: within each segment, the clean axis must stay close to
    # the real elapsed time. If a file drifted (e.g. really 49 Hz) the
    # difference would grow along the segment and trip this assert.
    seg_start = df.groupby("segment")["elapsed"].transform("first")
    deviation = np.abs((df["elapsed"] - seg_start) - df["t"])
    worst_ms = deviation.max() * 1000
    assert worst_ms < 5, f"{df['subject'].iloc[0]}: timing drift of {worst_ms:.1f} ms"
    print(f"    max deviation from a perfect 50 Hz grid: {worst_ms:.2f} ms")
    return df


def print_segment_stats(df: pd.DataFrame) -> None:
    """Print how many segments there are and how long they are."""
    seg_len = df.groupby("segment").size()
    fs = config.SAMPLE_RATE_HZ
    print(f"\nSegments total: {len(seg_len)}")
    print(f"Segment length in seconds: min {seg_len.min() / fs:.2f}, "
          f"median {seg_len.median() / fs:.1f}, max {seg_len.max() / fs:.1f}")
    too_short = (seg_len < config.WINDOW_SIZE).sum()
    print(f"Segments shorter than one window ({config.WINDOW_SIZE} samples): "
          f"{too_short} (these will simply produce no windows in step 5)")


def first_long_run(df: pd.DataFrame, label, min_rows: int) -> pd.DataFrame:
    """Return the first unbroken run of `label` with at least `min_rows` rows.

    Used for the signal plots. Rows of one label are usually not contiguous
    (other activities lie in between), so we need an unbroken *run* inside
    one segment. A new run starts whenever the segment or the label changes
    from one row to the next; cumsum turns those change points into a run
    number.
    """
    changed = (df["label"] != df["label"].shift()) | (df["segment"] != df["segment"].shift())
    run_id = changed.cumsum()
    run_len = run_id.map(run_id.value_counts())
    first = run_id[(df["label"] == label) & (run_len >= min_rows)].iloc[0]
    return df[run_id == first]
