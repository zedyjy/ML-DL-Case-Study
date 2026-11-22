# %% [markdown]
# # Step 3: load RealWorld HAR into the common table
#
# Input : `data/Real world HAR/s01 ... s15/acc_<activity>[_<n>]_<thigh|waist>.csv`
# Output: `processed/realworld.parquet`
#
# RealWorld is the messy one. Each recording is TWO files (thigh phone and
# waist phone) with their own clocks, in m/s², with phone orientation that
# differs between subjects and even between recordings. This script:
#
#  1. Finds the thigh + waist file pair for every (subject, activity,
#     recording number) and converts m/s² to g.
#  2. Repairs the clock of each file. The phones sampled steadily at
#     ~49.95 Hz, but Android stamped the samples in bursts (1-2 s holes
#     followed by samples 1 ms apart). So we trust the *sample count* and fit
#     a straight line "time = a + b * sample_index" through the raw
#     timestamps. A few files contain one real dropout (a permanent jump of
#     2-3 s); we detect those and split the file there.
#  3. Puts thigh and waist on one shared 50 Hz grid (linear interpolation).
#  4. Synchronises the two phones: their clocks disagree by up to 4 s. In
#     every walking / running / stairs recording we find the shift that best
#     lines up the footsteps seen by both phones (cross-correlation of the
#     acceleration magnitude) and apply the subject's median shift to all of
#     that subject's recordings.
#  5. Aligns the axes to the HARTH convention "gravity on -x while upright".
#     For upright activities each recording uses its own mean acceleration
#     as the gravity direction; sitting and lying reuse the rotation of the
#     subject's standing recording (because for them the mean direction is
#     not "down" relative to the sensor - that is the information we want
#     to keep). Rotation about the vertical axis stays unknown; the model
#     will be made robust to it in step 5/6 (decision 2, agreed 16 Nov 2026).
#  6. Saves the table with the activity name from the file name as label.

# %%
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import print_segment_stats

# %% [markdown]
# ## Settings for this step

# %%
FILE_PATTERN = re.compile(r"acc_([a-z]+)(?:_(\d))?_(thigh|waist)\.csv")

# Activities during which the body is upright, so the mean acceleration of
# the recording points "down" along the body and can be used as the gravity
# reference for axis alignment.
UPRIGHT = {"standing", "walking", "running", "climbingup", "climbingdown", "jumping"}

# Activities where thigh and waist both see footsteps -> we can synchronise
# the two phones by cross-correlation. For static activities the shift does
# not matter (nothing changes over time anyway).
DYNAMIC = {"walking", "running", "climbingup", "climbingdown", "jumping"}

MAX_SYNC_LAG_S = 15.0      # search window for the thigh/waist clock offset
CLOCK_STEP_MIN_S = 1.0     # a permanent timestamp jump bigger than this is a real dropout
CLOCK_STEP_WIN = 500       # samples (10 s) used for the before/after comparison
GRID_DT = 1.0 / config.SAMPLE_RATE_HZ

# %% [markdown]
# ## Helper 1: read one file and repair its clock
#
# `np.polyfit(index, time, 1)` fits the straight line that best explains the
# raw timestamps. Its slope is the true sample period (about 20.02 ms); the
# burst noise averages out. The residual `raw - fitted` looks like a
# sawtooth of +-1.6 s (buffering) - harmless. A *real* dropout shows up as a
# permanent step in that residual. We find steps by comparing the median
# residual 10 s before and 10 s after each point.

# %%
def fit_clock(t_raw: np.ndarray) -> np.ndarray:
    i = np.arange(len(t_raw))
    slope, intercept = np.polyfit(i, t_raw, 1)
    return intercept + slope * i


def find_clock_steps(t_raw: np.ndarray) -> list:
    """Indices where the timestamp makes a permanent jump (real dropouts)."""
    resid = t_raw - fit_clock(t_raw)
    win = CLOCK_STEP_WIN
    rolling_median = pd.Series(resid).rolling(win, center=True, min_periods=win // 2).median().to_numpy()
    # median after vs before, centred at index k + win/2
    jump = rolling_median[win:] - rolling_median[:-win]
    flagged = np.where(np.abs(jump) > CLOCK_STEP_MIN_S)[0]
    if len(flagged) == 0:
        return []
    # consecutive flagged indices belong to the same event -> keep the peak
    groups = np.split(flagged, np.where(np.diff(flagged) > 1)[0] + 1)
    return [int(g[np.argmax(np.abs(jump[g]))] + win // 2) for g in groups]


def read_realworld_file(path: Path) -> list:
    """Return a list of clock-repaired pieces: [(t_seconds, acc_in_g), ...].

    Usually one piece. More if the file contains a real dropout; the samples
    right around a dropout are discarded because their timing is uncertain.
    """
    df = pd.read_csv(path)
    t_raw = df["attr_time"].to_numpy() / 1000.0                 # ms -> s (Unix epoch)
    acc = df[["attr_x", "attr_y", "attr_z"]].to_numpy() / config.G_MS2  # m/s² -> g

    cuts = find_clock_steps(t_raw)
    if cuts:
        print(f"    {path.parent.name}/{path.name}: real dropout(s) at sample {cuts}, splitting")
    bounds = [0] + cuts + [len(t_raw)]
    margin = CLOCK_STEP_WIN // 2
    pieces = []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if k > 0:
            a += margin            # discard uncertain samples after a cut
        if k < len(bounds) - 2:
            b -= margin            # ... and before a cut
        if b - a < config.WINDOW_SIZE:
            continue
        pieces.append((fit_clock(t_raw[a:b]), acc[a:b]))
    return pieces


# %% [markdown]
# ## Helper 2: thigh + waist onto one 50 Hz grid, then synchronise
#
# Each phone's pieces are interpolated onto a common grid. Grid points that
# fall into a dropout of either phone are invalid; the remaining valid runs
# become the segments.

# %%
def interpolate_onto_grid(pieces, grid):
    """Linear interpolation of every axis onto `grid`; NaN outside the pieces."""
    out = np.full((len(grid), 3), np.nan, dtype=np.float32)
    for t, acc in pieces:
        inside = (grid >= t[0]) & (grid <= t[-1])
        for ax in range(3):
            out[inside, ax] = np.interp(grid[inside], t, acc[:, ax])
    return out


def _best_lag(a, b):
    """Lag (in samples) at which signal a best matches signal b."""
    corr = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")
    keep = np.abs(lags) <= int(MAX_SYNC_LAG_S / GRID_DT)
    return lags[keep][np.argmax(corr[keep])]


def sync_lag_seconds(thigh, waist):
    """Clock offset between the phones, from the acceleration magnitudes.

    Both phones feel every footstep, so the magnitude |a| of the two signals
    has the same rhythm. `correlate(thigh, waist)` is large at the lag k where
    thigh[n + k] matches waist[n], i.e. where the thigh signal runs k samples
    *behind* the waist signal. Shifting the waist clock by +k*dt lines them up.

    Returns (lag in seconds, spread in seconds). The spread is how much the
    lag differs when measured in the four quarters of the recording
    separately: a small spread means the estimate is trustworthy.
    """
    a = np.nan_to_num(np.linalg.norm(thigh, axis=1)); a -= a.mean()
    b = np.nan_to_num(np.linalg.norm(waist, axis=1)); b -= b.mean()
    lag = _best_lag(a, b) * GRID_DT
    n = len(a)
    quarters = [_best_lag(a[q * n // 4:(q + 1) * n // 4], b[q * n // 4:(q + 1) * n // 4]) * GRID_DT
                for q in range(4)]
    return lag, max(quarters) - min(quarters)


def combine_recording(thigh_pieces, waist_pieces, waist_offset_s=0.0):
    """Both phones on one 50 Hz grid; the waist clock shifted by `waist_offset_s`."""
    waist_pieces = [(t + waist_offset_s, acc) for t, acc in waist_pieces]
    # common time range of the two phones
    t0 = max(thigh_pieces[0][0][0], waist_pieces[0][0][0])
    t1 = min(thigh_pieces[-1][0][-1], waist_pieces[-1][0][-1])
    grid = np.arange(t0, t1, GRID_DT)

    thigh = interpolate_onto_grid(thigh_pieces, grid)
    waist = interpolate_onto_grid(waist_pieces, grid)
    valid = ~np.isnan(thigh).any(axis=1) & ~np.isnan(waist).any(axis=1)
    return grid, thigh, waist, valid


# The clock offset between the two phones changes from recording to
# recording (the phones re-sync their clocks in between), so every moving
# recording gets its own measured lag. Static recordings (standing, sitting,
# lying) have no footsteps to measure a lag from, but there the offset does
# not matter either; they simply get the subject's median lag.


# %% [markdown]
# ## Helper 3: axis alignment (gravity -> -x)
#
# `rotation_to_minus_x(g)` builds the smallest rotation that turns the
# direction `g` into (-1, 0, 0). This is the Rodrigues formula: rotate about
# the axis `g x target` by the angle between them. After applying it, the
# recording's mean acceleration while upright sits on -x, exactly like the
# HARTH sensors ("x = down" -> reads about -1 g).

# %%
def rotation_to_minus_x(g):
    g = g / np.linalg.norm(g)
    target = np.array([-1.0, 0.0, 0.0])
    v = np.cross(g, target)
    s = np.linalg.norm(v)          # sin(angle)
    c = float(np.dot(g, target))   # cos(angle)
    if s < 1e-8:                    # already (anti)parallel
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


# %% [markdown]
# ## Run it for all subjects

# %%
subjects = sorted(p for p in config.REALWORLD_DIR.glob("s*") if p.is_dir())
print(f"Found {len(subjects)} RealWorld subjects")

parts, log = [], []
for subj_dir in subjects:
    subj = subj_dir.name
    # collect files: key = (activity, recording number) -> {"thigh": path, "waist": path}
    recordings = {}
    for f in sorted(subj_dir.glob("acc_*.csv")):
        activity, rec, pos = FILE_PATTERN.match(f.name).groups()
        recordings.setdefault((activity, rec or "1"), {})[pos] = f

    # Gravity reference of this subject's standing recording, per position.
    # Used for sitting and lying, which have no upright reference of their own.
    standing_key = next(k for k in recordings if k[0] == "standing")
    standing_rot = {}
    for pos in ("thigh", "waist"):
        pieces = read_realworld_file(recordings[standing_key][pos])
        mean_g = np.concatenate([acc for _, acc in pieces]).mean(axis=0)
        standing_rot[pos] = rotation_to_minus_x(mean_g)

    # Pass 1: read every recording and measure the thigh/waist lag where possible.
    loaded, lags, spreads = {}, {}, {}
    for (activity, rec), files in sorted(recordings.items()):
        if len(files) < 2:
            print(f"  {subj} {activity}_{rec}: missing {set(['thigh', 'waist']) - files.keys()} -> skipped")
            continue
        thigh_pieces = read_realworld_file(files["thigh"])
        waist_pieces = read_realworld_file(files["waist"])
        loaded[(activity, rec)] = (thigh_pieces, waist_pieces)
        if activity in DYNAMIC:
            _, thigh, waist, _ = combine_recording(thigh_pieces, waist_pieces)
            lags[(activity, rec)], spreads[(activity, rec)] = sync_lag_seconds(thigh, waist)
    subject_median = float(np.median(list(lags.values())))

    # Pass 2: build the synchronised, aligned table.
    for (activity, rec), (thigh_pieces, waist_pieces) in loaded.items():
        lag = lags.get((activity, rec), subject_median)
        grid, thigh, waist, valid = combine_recording(thigh_pieces, waist_pieces, lag)

        # --- axis alignment -------------------------------------------------
        aligned = {}
        for pos, data in (("thigh", thigh), ("waist", waist)):
            if activity in UPRIGHT:
                R = rotation_to_minus_x(np.nanmean(data, axis=0))
            else:
                R = standing_rot[pos]
            ok = ~np.isnan(data).any(axis=1)      # skip dropout holes (NaN rows)
            # (np.errstate: numpy on Apple silicon raises spurious "divide by
            # zero in matmul" warnings for some array sizes; the result is fine)
            with np.errstate(all="ignore"):
                data[ok] = (data[ok].astype(np.float64) @ R.T).astype(np.float32)
            # Waist while sitting/lying should still read mostly "down" on -x.
            # +x means the phone was turned upside down between the standing
            # and this recording -> rotate 180 degrees about z.
            if pos == "waist" and activity == "sitting" and np.nanmean(data[:, 0]) > 0.3:
                print(f"    {subj} {activity}_{rec} waist: upside down vs standing, flipping")
                data[:, :2] *= -1          # x -> -x, y -> -y
            aligned[pos] = data
        thigh, waist = aligned["thigh"], aligned["waist"]

        # --- segments = runs of valid grid points ----------------------------
        change = np.diff(valid.astype(int), prepend=0)
        run_id = np.cumsum(change == 1)          # increases at every start of a valid run
        n_rows = 0
        for r in np.unique(run_id[valid]):
            idx = np.where((run_id == r) & valid)[0]
            if len(idx) < config.WINDOW_SIZE:
                continue
            seg = pd.DataFrame({
                "dataset": "realworld",
                "subject": f"rw_{subj}",
                "segment": f"rw_{subj}_{activity}{rec}_{r:02d}",
                "t": (np.arange(len(idx)) * GRID_DT).astype("float32"),
                "back_x": waist[idx, 0], "back_y": waist[idx, 1], "back_z": waist[idx, 2],
                "thigh_x": thigh[idx, 0], "thigh_y": thigh[idx, 1], "thigh_z": thigh[idx, 2],
                "label": activity,
            })
            parts.append(seg)
            n_rows += len(idx)

        log.append(dict(subject=subj, activity=activity, rec=rec, rows=n_rows,
                        minutes=round(n_rows / 50 / 60, 1),
                        sync_lag_s=round(lag, 2), lag_spread_s=spreads.get((activity, rec), np.nan),
                        **{f"thigh_{c}": v for c, v in zip("xyz", np.nanmean(thigh, axis=0).round(2))},
                        **{f"waist_{c}": v for c, v in zip("xyz", np.nanmean(waist, axis=0).round(2))}))
    print(f"  {subj}: {len([k for k in recordings if len(recordings[k]) == 2])} recordings")

rw = pd.concat(parts, ignore_index=True)
for c in config.SENSOR_COLS:
    rw[c] = rw[c].astype("float32")
log = pd.DataFrame(log)

# %% [markdown]
# ## Sanity checks

# %%
print("\nShape:", rw.shape)
print("Subjects:", rw["subject"].nunique(), " recordings:", len(log))
print("\nMinutes per activity:")
print((rw.groupby("label").size() / 50 / 60).round(1).to_string())
print_segment_stats(rw)

# %%
# Phone synchronisation. Left: the waist clock shift applied to each moving
# recording (seconds; separate phones, so several seconds is normal).
# Right: how much the four quarters of the recording disagree about that
# shift. Walking is periodic (one step about every 0.5 s), so a spread of
# about one step means the correlation could not tell neighbouring steps
# apart; anything much larger means the estimate is unreliable.
pd.set_option("display.width", 200)
print("\nWaist clock shift applied (s), per moving recording:")
print(log.pivot_table(index="subject", columns="activity", values="sync_lag_s", aggfunc="first")
      .drop(columns=["standing", "sitting", "lying"]).round(2).to_string())
print("\nQuarter-to-quarter spread of that shift (s):")
print(log.pivot_table(index="subject", columns="activity", values="lag_spread_s", aggfunc="first")
      .round(2).to_string())
unreliable = log[log["lag_spread_s"] > 1.0]
print(f"\nRecordings with spread > 1 s: {len(unreliable)}")
if len(unreliable):
    print(unreliable[["subject", "activity", "rec", "sync_lag_s", "lag_spread_s"]].to_string(index=False))

# %%
# Orientation after alignment. Upright activities must show thigh_x and
# waist_x near -1 and y, z near 0 (by construction). Sitting should look like
# HARTH sitting: thigh_x near 0 (thigh horizontal), waist_x still near -1.
print("\nMean acceleration after alignment (g), per recording:")
cols = ["subject", "activity", "rec", "thigh_x", "thigh_y", "thigh_z", "waist_x", "waist_y", "waist_z"]
print(log[cols].to_string(index=False))

# %%
# Plot 1: before/after alignment is in processed/plots/03_realworld_alignment_preview.png
# (made during exploration). Plot 2 here: a walking recording after
# synchronisation - the thigh and waist magnitude peaks (footsteps) should
# line up - next to HARTH walking for comparison.
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
harth = pd.read_parquet(config.PROCESSED_DIR / "harth.parquet", filters=[("subject", "==", "harth_S008")])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
g = rw[rw["segment"].str.startswith("rw_s01_walking1")].iloc[5000:5500]
h = harth[harth["label"] == 1].iloc[20000:20500]
t = np.arange(500) / 50
for ax, d, name in [(axes[0], g, "RealWorld s01 walking (aligned + synchronised)"),
                    (axes[1], h, "HARTH S008 walking")]:
    ax.plot(t, d["thigh_x"].to_numpy(), label="thigh_x")
    ax.plot(t, d["back_x"].to_numpy(), label="back_x / waist_x", alpha=0.7)
    ax.set_title(name); ax.set_ylabel("acceleration [g]"); ax.legend(loc="upper right")
axes[-1].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(config.PLOTS_DIR / "03_realworld_vs_harth_walking.png", dpi=120)
if "ipykernel" in sys.modules:
    plt.show()

# %% [markdown]
# ## Save

# %%
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.PROCESSED_DIR / "realworld.parquet"
rw.to_parquet(out_path, index=False)
log.to_csv(config.PROCESSED_DIR / "realworld_recording_log.csv", index=False)
print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved per-recording log to processed/realworld_recording_log.csv")
print(rw.head())
