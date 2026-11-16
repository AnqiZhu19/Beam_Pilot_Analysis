"""Loading, trial table, clock sync and per-trial resampling for the new data format."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import config_v2 as C

sys.path.insert(0, C.OLD_CODE_DIR)
import data_utils as DU  # noqa: E402  (old pilot_analysis helpers, unchanged)

EVENT_TIME = "time since start of unity"
LOG_COLS = ["Time", "MyPosX", "MyPosY", "OtherPosX", "OtherPosY"]


def load(folder):
    trial = pd.read_csv(os.path.join(folder, "trial.csv"), encoding="utf-8-sig")
    trial.columns = [c.strip() for c in trial.columns]
    for col in ["primary in target", "secondary in target"]:
        trial[col] = trial[col].astype(str).str.strip().str.lower().eq("true")
    for col in ["condition", "phase", "result"]:
        trial[col] = trial[col].astype(str).str.strip()

    logs = []
    for name in ["primarySpringCouplingLog.csv", "secondarySpringCouplingLog.csv"]:
        df = pd.read_csv(os.path.join(folder, name), encoding="utf-8-sig",
                         usecols=lambda c: c.strip() in LOG_COLS)
        df.columns = [c.strip() for c in df.columns]
        df = (df.dropna(subset=["Time"]).sort_values("Time")
                .drop_duplicates(subset="Time", keep="first").reset_index(drop=True))
        logs.append(df)
    return trial, logs[0], logs[1]


def _last_entry(g, col):
    """Time (s from trial begin) of the last False->True transition of an
    in-target flag, i.e. when the player entered the target for good."""
    f = g[col].values
    t = g["time since start of trial"].values
    if not f[-1]:
        return np.nan
    idx = np.where(f & ~np.r_[False, f[:-1]])[0]
    return float(t[idx[-1]]) if len(idx) else np.nan


def build_trial_table(trial):
    """One row per (block, trial): begin/end on the trial.csv clock, result,
    final target-entry times. All blocks (incl. failed trials) are kept here;
    filtering happens in select_trials()."""
    rows = []
    for (b, tr), g in trial.groupby(["block num", "trial num"], sort=False):
        g = g.sort_values(EVENT_TIME)
        beg = g.loc[g["phase"] == "begin", EVENT_TIME]
        end = g.loc[g["phase"] == "end", EVENT_TIME]
        res = g.loc[g["phase"] == "end", "result"]
        t_begin = float(beg.iloc[0]) if len(beg) else float(g[EVENT_TIME].min())
        t_end = float(end.iloc[0]) if len(end) else float(g[EVENT_TIME].max())
        tp, ts = _last_entry(g, "primary in target"), _last_entry(g, "secondary in target")
        rows.append(dict(
            block_num=int(b), trial=int(tr), condition=g["condition"].iloc[0],
            result=res.iloc[0] if len(res) else "none",
            t_begin=t_begin, t_end=t_end, trial_duration=t_end - t_begin,
            target_entry_P1=tp, target_entry_P2=ts,
            # > 0: P1 entered the target first
            target_entry_diff=ts - tp,
        ))
    tt = pd.DataFrame(rows).sort_values("t_begin").reset_index(drop=True)
    tt["block"] = tt["block_num"].map(C.BLOCKS)
    return tt


def select_trials(tt):
    tt = tt[tt["block"].notna()].copy()
    if C.SUCCESS_ONLY:
        tt = tt[tt["result"] == "success"].copy()
    tt["trial_in_block"] = tt.groupby("block").cumcount() + 1   # among kept trials
    tt["n_in_block"] = tt.groupby("block")["trial"].transform("size")
    n = np.minimum(C.EARLY_N, tt["n_in_block"] // 2)
    tt["early_late"] = np.where(tt["trial_in_block"] <= n, "early",
                                np.where(tt["trial_in_block"] > tt["n_in_block"] - n, "late", "mid"))
    tt["global_idx"] = np.arange(1, len(tt) + 1)
    return tt.reset_index(drop=True)


def sync(tt, p, s):
    """Shift the logs onto the trial.csv clock (old detect_time_offset)."""
    offset, n_match, n_win, spread = DU.detect_time_offset(tt, p)
    p["Time"] -= offset
    s["Time"] -= offset
    return dict(offset=offset, n_matched=n_match, n_windows=n_win,
                n_trials=len(tt), spread=spread)


_SOS = butter(C.FILTER_ORDER, C.LOWPASS_HZ / (C.FS / 2), output="sos")


def lowpass(x):
    x = np.asarray(x, float)
    if len(x) < 100:
        return x
    return sosfiltfilt(_SOS, x)


def onset_offset(speed, t, peak=None):
    """First / last sample of a sustained run above max(frac*peak, floor).
    `peak` lets both hands share one threshold (default: this signal's peak)."""
    speed = np.asarray(speed, float).copy()
    e = int(0.03 * C.FS)
    speed[:e] = 0
    speed[-e:] = 0
    thr = max(C.ONSET_FRAC * (np.nanmax(speed) if peak is None else peak), C.ONSET_MIN)
    above = speed > thr
    edges = np.diff(np.r_[0, above.astype(int), 0])
    st, sp = np.where(edges == 1)[0], np.where(edges == -1)[0]
    keep = (sp - st) >= max(1, int(C.ONSET_SUSTAIN_S * C.FS))
    if not keep.any():
        return np.nan, np.nan
    return float(t[st[keep][0]]), float(t[sp[keep][-1] - 1])


def resample_trial(row, p, s):
    """Both players on one uniform FS grid over [t_begin, t_end]; filtered
    kinematics and dyadic signals. Returns (DataFrame or None, qc dict)."""
    t0, t1 = row["t_begin"], row["t_end"]
    pm = p[(p["Time"] >= t0 - 0.05) & (p["Time"] <= t1 + 0.05)]
    sm = s[(s["Time"] >= t0 - 0.05) & (s["Time"] <= t1 + 0.05)]
    qc = dict(n_p=len(pm), n_s=len(sm))
    if len(pm) < 50 or len(sm) < 50:
        qc["coverage"] = 0.0
        return None, qc
    ta = max(t0, pm["Time"].min(), sm["Time"].min())
    tb = min(t1, pm["Time"].max(), sm["Time"].max())
    t = np.arange(ta, tb, 1.0 / C.FS)
    qc["coverage"] = (tb - ta) / max(t1 - t0, 1e-9)
    qc["max_gap"] = float(max(np.diff(pm["Time"].values).max(), np.diff(sm["Time"].values).max()))
    if qc["coverage"] < C.MIN_COVERAGE or len(t) < 100:
        return None, qc

    d = pd.DataFrame({"t": t - t0})
    d["x1"] = DU._interp(pm, "MyPosX", t)
    d["x2"] = DU._interp(sm, "MyPosX", t)
    d["y1"] = DU._interp(pm, "MyPosY", t)
    d["y2"] = DU._interp(pm, "OtherPosY", t)          # partner height as seen by the Primary
    y2_own = DU._interp(sm, "MyPosY", t)
    x2_other = DU._interp(pm, "OtherPosX", t)
    qc["y2_rms_diff_mm"] = 1e3 * float(np.sqrt(np.mean((d["y2"] - y2_own) ** 2)))
    qc["x2_rms_diff_mm"] = 1e3 * float(np.sqrt(np.mean((d["x2"] - x2_other) ** 2)))

    dt = 1.0 / C.FS
    for k in ("x1", "x2", "y1", "y2"):
        d[k] = lowpass(d[k].values)
        d["v" + k] = np.gradient(d[k].values, dt)
    d["dx"] = d["x1"] - d["x2"]
    d["dy"] = d["y1"] - d["y2"]
    d["obj_x"] = 0.5 * (d["x1"] + d["x2"])
    d["obj_v"] = 0.5 * (d["vx1"] + d["vx2"])
    d["theta"] = np.degrees(np.arcsin(np.clip(d["dy"] / C.BEAM_LENGTH_M, -1, 1)))
    d["theta_dot"] = np.gradient(d["theta"].values, dt)
    return d, qc
