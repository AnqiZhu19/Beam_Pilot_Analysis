"""
Loading, clock synchronisation, phase labelling and trial segmentation for the
dyadic beam-balancing task.

Raw files (one folder per dyad)
-------------------------------
trial.csv  - one row per event:
    'time since start of unity', 'time since start of trial', 'condition index',
    'block num', 'trial num', 'condition', 'phase', 'result',
    'primary in target', 'secondary in target'
    phase in {begin, primary_left_home, secondary_left_home, running, end}

primarySpringCouplingLog.csv / secondarySpringCouplingLog.csv - ~500 Hz:
    Time, Robot, MyPosX/Y/Z, OtherPosX/Y/Z, RelativePosX, RelativeVelX,
    ToolForceX/Y/Z, F1, F2, SpringForce, HapticForce

Conventions used everywhere downstream
--------------------------------------
P1 = Primary player, P2 = Secondary player.
x  = movement axis (start -> target).   y = vertical axis.
dx = x1 - x2 (P1 ahead of P2 if > 0 in the +x direction).
tilt = asin((y1 - y2) / BEAM_LENGTH)  [deg], > 0 when P1's end is higher.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config as C

EVENT_TIME = "time since start of unity"


# ======================================================================
# LOADING
# ======================================================================
def _read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw(folder):
    trial = _read_csv(os.path.join(folder, "trial.csv"))
    p = _read_csv(os.path.join(folder, "primarySpringCouplingLog.csv"))
    s = _read_csv(os.path.join(folder, "secondarySpringCouplingLog.csv"))

    for col in ["primary in target", "secondary in target"]:
        trial[col] = trial[col].astype(str).str.strip().str.lower().eq("true")
    for col in ["condition", "phase"]:
        trial[col] = trial[col].astype(str).str.strip()

    for df in (p, s):
        for c in df.columns:
            if c != "Robot":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["Time"], inplace=True)
        df.sort_values("Time", inplace=True)
        df.drop_duplicates(subset="Time", keep="first", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return trial, p, s


# ======================================================================
# TRIAL TABLE + PHASES
# ======================================================================
def build_trial_table(trial):
    """One row per trial, chronological. Times are on the trial.csv clock."""
    trial = trial[~trial["condition"].isin(C.NON_TRIAL_CONDITIONS)]
    rows = []
    for (b, t), g in trial.groupby(["block num", "trial num"], sort=False):
        g = g.sort_values(EVENT_TIME)

        def first(phase):
            v = g.loc[g["phase"] == phase, EVENT_TIME]
            return float(v.iloc[0]) if len(v) else np.nan

        t_begin, t_end = first("begin"), first("end")
        if np.isnan(t_begin):
            t_begin = float(g[EVENT_TIME].min())
        if np.isnan(t_end):
            t_end = float(g[EVENT_TIME].max())
        t_p_home, t_s_home = first("primary_left_home"), first("secondary_left_home")
        t_p_tgt = g.loc[g["primary in target"], EVENT_TIME].min()
        t_s_tgt = g.loc[g["secondary in target"], EVENT_TIME].min()
        res = g.loc[g["phase"] == "end", "result"]

        homes = [v for v in (t_p_home, t_s_home) if not np.isnan(v)]
        t_move = min(homes) if homes else np.nan

        rows.append(dict(
            block=b, trial=t, condition=g["condition"].iloc[0],
            result=res.iloc[0] if len(res) else np.nan,
            t_begin=t_begin, t_end=t_end, t_move_start_event=t_move,
            t_running=first("running"),
            t_prim_home=t_p_home, t_sec_home=t_s_home,
            t_prim_target=t_p_tgt, t_sec_target=t_s_tgt,
            trial_duration=t_end - t_begin,
            # event-based lags, Secondary - Primary  (> 0 : Primary first)
            init_time_diff=t_s_home - t_p_home,
            target_time_diff=t_s_tgt - t_p_tgt,
        ))
    tt = pd.DataFrame(rows).sort_values("t_begin").reset_index(drop=True)
    tt["global_trial_idx"] = np.arange(1, len(tt) + 1)
    tt["trial_in_block"] = tt.groupby("block").cumcount() + 1
    return tt


def assign_phases(tt, protocol):
    """Label contiguous runs of the same condition (chronological order).
    Baseline runs become 'Baseline 1', 'Baseline 2', ... . Checks the result
    against the expected protocol and warns on mismatch."""
    tt = tt.copy()
    run_id = (tt["condition"] != tt["condition"].shift()).cumsum()
    labels, seen = {}, {}
    for rid, g in tt.groupby(run_id, sort=True):
        cond = g["condition"].iloc[0]
        seen[cond] = seen.get(cond, 0) + 1
        if cond.lower().startswith("baseline"):
            labels[rid] = f"Baseline {seen[cond]}"
        else:
            labels[rid] = cond if seen[cond] == 1 else f"{cond} ({seen[cond]})"
    tt["phase"] = run_id.map(labels)
    tt["phase_idx"] = run_id.values
    tt["trial_in_phase"] = tt.groupby("phase").cumcount() + 1
    tt["n_in_phase"] = tt.groupby("phase")["trial"].transform("size")

    found = list(dict.fromkeys(tt["phase"]))
    expected = C.PROTOCOLS[protocol]["phases"]
    ok = found == expected
    if not ok:
        warnings.warn(
            f"Phase sequence {found} does not match protocol {protocol} "
            f"{expected}. Check the PILOTS entry in config.py.")
    tt["early_late"] = np.where(
        tt["trial_in_phase"] <= C.EARLY_N, "early",
        np.where(tt["trial_in_phase"] > tt["n_in_phase"] - C.LATE_N, "late", "mid"))
    return tt, found, ok


# ======================================================================
# CLOCK SYNCHRONISATION
# ======================================================================
def detect_time_offset(tt, p, gap_threshold=1.0, tol=0.25):
    """The coupling logs only record during trials and may run on a clock
    that is offset from trial.csv. Every log-window start is a candidate
    offset; the one that aligns the most trial starts (within `tol` s) wins
    (robust to missing/extra windows, unlike pairing windows by index).
    Returns (offset, n_matched, n_windows, spread)."""
    times = p["Time"].values
    gaps = np.where(np.diff(times) > gap_threshold)[0]
    win = np.concatenate(([times[0]], times[gaps + 1]))
    tb = tt["t_begin"].values

    cands = np.unique(np.round(np.concatenate([[0.0], (win[:, None] - tb[None, :]).ravel()]), 3))
    best, best_n = 0.0, -1
    for c in cands:
        d = np.abs(win[None, :] - (tb[:, None] + c)).min(axis=1)
        n = int((d < tol).sum())
        if n > best_n:
            best, best_n = c, n
    # refine with the median residual of the matched trials
    idx = np.abs(win[None, :] - (tb[:, None] + best)).argmin(axis=1)
    resid = win[idx] - tb
    m = np.abs(resid - best) < tol
    offset = float(np.median(resid[m])) if m.any() else float(best)
    spread = float(np.std(resid[m])) if m.sum() > 1 else np.nan

    # if logging is continuous (no gaps) the window method cannot work:
    # check instead that the logs simply cover the trials on the same clock
    if len(win) == 1:
        offset = 0.0 if (times[0] <= tb.min() + tol) else float(best)
    return offset, best_n, len(win), spread


def apply_time_offset(p, s, offset):
    p["Time"] = p["Time"] - offset
    s["Time"] = s["Time"] - offset


# ======================================================================
# SEGMENTATION
# ======================================================================
def _lowpass(x, fs=C.FS, fc=C.LOWPASS_HZ):
    b, a = butter(4, fc / (fs / 2.0))
    if len(x) <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x)


def _interp(df, col, t):
    if col not in df.columns:
        return np.full(len(t), np.nan)
    y = df[col].values
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return np.full(len(t), np.nan)
    return np.interp(t, df["Time"].values[ok], y[ok])


def segment_trial(row, p, s):
    """Resample both players onto one uniform FS grid for [t_begin, t_end],
    low-pass, differentiate and derive dyadic signals.
    Returns (DataFrame, qc dict) or (None, qc) if the trial has no data."""
    t0, t1 = row["t_begin"], row["t_end"]
    pm = p[(p["Time"] >= t0 - 0.05) & (p["Time"] <= t1 + 0.05)]
    sm = s[(s["Time"] >= t0 - 0.05) & (s["Time"] <= t1 + 0.05)]
    qc = dict(n_p=len(pm), n_s=len(sm))
    if len(pm) < 20 or len(sm) < 20:
        return None, qc

    t = np.arange(max(t0, pm["Time"].min(), sm["Time"].min()),
                  min(t1, pm["Time"].max(), sm["Time"].max()), 1.0 / C.FS)
    if len(t) < 20:
        return None, qc
    qc["max_gap_p"] = float(np.diff(pm["Time"].values).max())
    qc["max_gap_s"] = float(np.diff(sm["Time"].values).max())
    qc["coverage"] = (t[-1] - t[0]) / max(t1 - t0, 1e-9)

    d = pd.DataFrame({"t": t - t0})
    d["x1"] = _interp(pm, "MyPosX", t)
    d["y1"] = _interp(pm, "MyPosY", t)
    x2_s, x2_p = _interp(sm, "MyPosX", t), _interp(pm, "OtherPosX", t)
    y2_s, y2_p = _interp(sm, "MyPosY", t), _interp(pm, "OtherPosY", t)
    d["x2"] = x2_s if C.SECONDARY_X_SOURCE == "secondary_file" else x2_p
    d["y2"] = y2_p if C.SECONDARY_Y_SOURCE == "primary_other" else y2_s
    # QC: do the two possible sources of the partner's position agree?
    if np.isfinite(x2_p).all() and np.isfinite(x2_s).all():
        qc["x2_rms_diff"] = float(np.sqrt(np.mean((x2_s - x2_p) ** 2)))
    if np.isfinite(y2_p).all() and np.isfinite(y2_s).all():
        qc["y2_rms_diff"] = float(np.sqrt(np.mean((y2_s - y2_p) ** 2)))

    for col, src, name in [("F1", pm, "F1"), ("F2", pm, "F2"),
                           ("Fh1", pm, "HapticForce"), ("Fh2", sm, "HapticForce"),
                           ("relx_logged", pm, "RelativePosX"),
                           ("relv_logged", pm, "RelativeVelX")]:
        d[col] = _interp(src, name, t)

    # filtered kinematics and derivatives (zero-phase, so no time shift
    # between players -> safe for lead-lag analysis)
    dt = 1.0 / C.FS
    for k in ("x1", "x2", "y1", "y2"):
        f = _lowpass(d[k].values)
        d[k + "_f"] = f
        v = np.gradient(f, dt)
        d["v" + k] = v
        d["a" + k] = np.gradient(v, dt)
    d.rename(columns={"vx1": "v1", "vx2": "v2", "ax1": "a1", "ax2": "a2"}, inplace=True)

    d["dx"] = d["x1_f"] - d["x2_f"]
    d["dv"] = d["v1"] - d["v2"]
    d["obj_v"] = 0.5 * (d["v1"] + d["v2"])          # beam-centre velocity
    d["tilt"] = np.degrees(np.arcsin(np.clip((d["y1_f"] - d["y2_f"]) / C.BEAM_LENGTH_M, -1, 1)))
    d["tilt_rate"] = np.gradient(d["tilt"].values, dt)

    if "relx_logged" in d and np.isfinite(d["relx_logged"]).all():
        qc["relx_rms_vs_x1_minus_x2"] = float(
            np.sqrt(np.mean((d["relx_logged"] - (d["x1"] - d["x2"])) ** 2)))

    # movement window: first home-leave event, else kinematic onset
    tm = row.get("t_move_start_event", np.nan)
    d.attrs["t_move"] = (tm - t0) if np.isfinite(tm) else np.nan
    return d, qc


def movement_onset_offset(speed, t, sustain_s=0.05, edge_s=0.05):
    """Onset / offset = first / last sample of a run where speed stays above
    max(frac*peak, floor) for >= sustain_s. The first/last edge_s of the
    window are ignored (filter edge transients)."""
    speed = np.asarray(speed, float).copy()
    e = int(edge_s * C.FS)
    if len(speed) > 4 * e:
        speed[:e] = 0
        speed[-e:] = 0
    thr = max(C.ONSET_SPEED_FRAC * np.nanmax(speed), C.ONSET_SPEED_MIN)
    above = speed > thr
    n = max(1, int(sustain_s * C.FS))
    # runs of consecutive True values
    edges = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    keep = (stops - starts) >= n
    if not keep.any():
        return np.nan, np.nan, None
    i0, i1 = starts[keep][0], stops[keep][-1]
    return t[i0], t[i1 - 1], (i0, i1)


def resample_normalized(t, y, t_start, t_stop, n=C.N_GRID):
    """Interpolate y(t) on n points between t_start and t_stop (0..100 %)."""
    if not (np.isfinite(t_start) and np.isfinite(t_stop)) or t_stop <= t_start:
        return np.full(n, np.nan)
    return np.interp(np.linspace(t_start, t_stop, n), t, np.asarray(y, float))
