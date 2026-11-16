"""
Kinematic metrics for the dyadic beam-balancing task.

Sign conventions (all "lead" metrics):  > 0  means  Player 1 (Primary) leads.
"""

import numpy as np
from scipy.signal import find_peaks

import config as C
from data_utils import movement_onset_offset, resample_normalized


# ======================================================================
# SMOOTHNESS - SPARC
# ======================================================================
def sparc(speed, fs=C.FS, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (Balasubramanian et al., 2012/2015).
    Closer to 0 (less negative) = smoother. Typical reaching: -1.4 to -3+."""
    speed = np.asarray(speed, float)
    speed = speed[np.isfinite(speed)]
    if len(speed) < 10 or np.max(np.abs(speed)) <= 0:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / Mf.max()
    sel = f <= fc
    f_sel, M_sel = f[sel], Mf[sel]
    above = np.nonzero(M_sel >= amp_th)[0]
    if len(above) < 2:
        return np.nan
    f_sel = f_sel[above[0]: above[-1] + 1]
    M_sel = M_sel[above[0]: above[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return -float(np.sum(np.sqrt(df ** 2 + np.diff(M_sel) ** 2)))


def n_submovements(speed, fs=C.FS, prominence_frac=0.10, min_sep_s=0.10):
    """Number of speed peaks (submovements / corrections)."""
    speed = np.asarray(speed, float)
    if len(speed) < 5 or not np.isfinite(speed).any():
        return np.nan
    pk, _ = find_peaks(speed, prominence=prominence_frac * np.nanmax(speed),
                       distance=max(1, int(min_sep_s * fs)))
    return int(len(pk))


# ======================================================================
# LEAD-LAG
# ======================================================================
def xcorr_lead_lag(a, b, fs=C.FS, max_lag_s=C.XCORR_MAX_LAG_S):
    """Normalised cross-correlation between two signals.
    Returns (lag_s, r_peak). lag > 0  <=>  b(t) ~ a(t - lag), i.e. `a` LEADS.
    Sub-sample precision by parabolic interpolation of the peak."""
    a = np.asarray(a, float) - np.nanmean(a)
    b = np.asarray(b, float) - np.nanmean(b)
    n = len(a)
    L = int(round(max_lag_s * fs))
    if n < 2 * L + 10 or np.std(a) == 0 or np.std(b) == 0:
        L = max(1, n // 4)
    lags = np.arange(-L, L + 1)
    r = np.empty(len(lags))
    for i, k in enumerate(lags):
        if k >= 0:
            x, y = a[: n - k], b[k:]
        else:
            x, y = a[-k:], b[: n + k]
        if len(x) < 5 or x.std() == 0 or y.std() == 0:
            r[i] = np.nan
        else:
            r[i] = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(r).any():
        return np.nan, np.nan
    i = int(np.nanargmax(r))
    lag = float(lags[i])
    if 0 < i < len(r) - 1 and np.isfinite(r[i - 1]) and np.isfinite(r[i + 1]):
        den = r[i - 1] - 2 * r[i] + r[i + 1]
        if den != 0:
            lag += 0.5 * (r[i - 1] - r[i + 1]) / den
    return lag / fs, float(r[i])


def windowed_lead_lag(a, b, fs=C.FS, win_s=C.WIN_LEADLAG_S, step_s=C.WIN_STEP_S,
                      max_lag_s=0.2, min_r=C.WIN_MIN_CORR):
    """Time-resolved leadership: lag in sliding windows.
    Returns (frac_a_leads, n_switches, lags array)."""
    w, st = int(win_s * fs), max(1, int(step_s * fs))
    lags = []
    for s0 in range(0, len(a) - w + 1, st):
        lag, r = xcorr_lead_lag(a[s0:s0 + w], b[s0:s0 + w], fs, max_lag_s)
        if np.isfinite(lag) and r >= min_r:
            lags.append(lag)
    lags = np.array(lags)
    if len(lags) == 0:
        return np.nan, np.nan, lags
    sgn = np.sign(lags[np.abs(lags) > 1.0 / fs])   # ignore ~0 lags
    frac = float(np.mean(sgn > 0)) if len(sgn) else np.nan
    switches = int(np.sum(np.diff(sgn) != 0)) if len(sgn) > 1 else 0
    return frac, switches, lags


# ======================================================================
# ENERGY / FORCE-BASED LEADERSHIP (from kinematics)
# ======================================================================
def energy_leader_index(f_on1, v1, v2, dt):
    """Energy transferred through the coupling.
    f_on1 = coupling force acting on P1 (on P2 it is -f_on1).
    E_i  = energy the coupling *delivers* to player i (being dragged along).
    Index = (E2 - E1) / (E1 + E2) in [-1, 1]; > 0  =>  P2 is dragged => P1 leads.
    With a pure spring f_on1 = -K*(x1 - x2) the index does not depend on K,
    so it can be computed from positions alone."""
    p1 = f_on1 * v1
    p2 = -f_on1 * v2
    e1 = np.nansum(np.clip(p1, 0, None)) * dt
    e2 = np.nansum(np.clip(p2, 0, None)) * dt
    tot = e1 + e2
    return (e2 - e1) / tot if tot > 0 else np.nan


def corrective_tilt_metrics(d, m=C.M_EFF_KG, deadband_deg=0.5):
    """Who corrects the beam tilt?  tilt > 0  <=>  P1's end is higher.
    P1 corrects by moving/accelerating DOWN, P2 by moving/accelerating UP.
      corr_force_i  : impulse of the tilt-reducing vertical kinematic force
                      (m * a_y), integrated over time  [N*s with m in kg]
      corr_vel_i    : tilt-reducing vertical displacement (integral of v_y)
    Share index (i1 - i2)/(i1 + i2): > 0  =>  P1 does more of the correcting."""
    dt = 1.0 / C.FS
    tilt = d["tilt"].values
    s = np.where(np.abs(tilt) > deadband_deg, np.sign(tilt), 0.0)
    f1 = np.clip(-m * d["ay1"].values * s, 0, None)
    f2 = np.clip(+m * d["ay2"].values * s, 0, None)
    u1 = np.clip(-d["vy1"].values * s, 0, None)
    u2 = np.clip(+d["vy2"].values * s, 0, None)
    I1, I2 = f1.sum() * dt, f2.sum() * dt
    D1, D2 = u1.sum() * dt, u2.sum() * dt
    out = dict(corr_force_impulse_P1=I1, corr_force_impulse_P2=I2,
               corr_disp_P1=D1, corr_disp_P2=D2)
    out["corr_force_share_idx"] = (I1 - I2) / (I1 + I2) if (I1 + I2) > 0 else np.nan
    out["corr_disp_share_idx"] = (D1 - D2) / (D1 + D2) if (D1 + D2) > 0 else np.nan
    return out


def n_tilt_corrections(tilt, min_amp_deg=1.0):
    """Number of tilt reversals (local extrema of tilt with >= min_amp
    excursion) = discrete balance corrections."""
    tilt = np.asarray(tilt, float)
    pk_hi, _ = find_peaks(tilt, prominence=min_amp_deg)
    pk_lo, _ = find_peaks(-tilt, prominence=min_amp_deg)
    return int(len(pk_hi) + len(pk_lo))


# ======================================================================
# PER-TRIAL METRICS
# ======================================================================
def trial_metrics(d):
    """All scalar metrics for one segmented trial (DataFrame from
    data_utils.segment_trial). Returns (metrics dict, (t_start, t_stop))."""
    t = d["t"].values
    dt = 1.0 / C.FS
    out = {}

    # --- movement window ---------------------------------------------------
    sp1, sp2 = np.abs(d["v1"].values), np.abs(d["v2"].values)
    on1, off1, w1 = movement_onset_offset(sp1, t)
    on2, off2, w2 = movement_onset_offset(sp2, t)
    t_move = d.attrs.get("t_move", np.nan)
    if not np.isfinite(t_move):
        t_move = np.nanmin([on1, on2])
    t_stop = t[-1]
    mv = (t >= t_move) & (t <= t_stop) if np.isfinite(t_move) else np.ones_like(t, bool)
    out["movement_time"] = t_stop - t_move if np.isfinite(t_move) else np.nan

    # --- per player kinematics ---------------------------------------------
    for i, sp, on, off, w in [(1, sp1, on1, off1, w1), (2, sp2, on2, off2, w2)]:
        out[f"onset_P{i}"], out[f"offset_P{i}"] = on, off
        if w is None:
            continue
        seg = sp[w[0]:w[1]]
        k = int(np.argmax(sp))
        out[f"peak_speed_P{i}"] = float(sp[k])
        out[f"t_peak_speed_P{i}"] = float(t[k])
        out[f"mean_speed_P{i}"] = float(np.mean(seg))
        out[f"sparc_P{i}"] = sparc(seg)
        out[f"n_submov_P{i}"] = n_submovements(seg)
        x = d[f"x{i}_f"].values
        out[f"path_len_P{i}"] = float(np.sum(np.abs(np.diff(x[w[0]:w[1]]))))
        out[f"rms_kin_force_P{i}"] = float(np.sqrt(np.mean((C.M_EFF_KG * d[f"a{i}"].values[w[0]:w[1]]) ** 2)))

    out["sparc_object"] = sparc(np.abs(d["obj_v"].values[mv]))
    out["onset_lag_kin"] = out.get("onset_P2", np.nan) - out.get("onset_P1", np.nan)
    out["peak_speed_lag"] = out.get("t_peak_speed_P2", np.nan) - out.get("t_peak_speed_P1", np.nan)

    # --- spatial coordination ---------------------------------------------
    dx = d["dx"].values[mv]
    out["mean_dx"] = float(np.mean(dx))
    out["mean_abs_dx"] = float(np.mean(np.abs(dx)))
    out["peak_abs_dx"] = float(np.max(np.abs(dx)))
    out["rms_dx"] = float(np.sqrt(np.mean(dx ** 2)))
    v1m, v2m = d["v1"].values[mv], d["v2"].values[mv]
    thr = C.ONSET_SPEED_MIN
    moving = (np.abs(v1m) > thr) & (np.abs(v2m) > thr)
    out["frac_opposing_motion"] = float(np.mean(v1m[moving] * v2m[moving] < 0)) if moving.any() else np.nan

    # --- beam tilt -----------------------------------------------------------
    tl = d["tilt"].values[mv]
    out["max_abs_tilt"] = float(np.max(np.abs(tl)))
    out["rms_tilt"] = float(np.sqrt(np.mean(tl ** 2)))
    out["mean_tilt"] = float(np.mean(tl))
    out["sd_tilt"] = float(np.std(tl))
    out["pct_time_level"] = 100.0 * float(np.mean(np.abs(tl) <= C.TILT_TOLERANCE_DEG))
    out["n_tilt_corrections"] = n_tilt_corrections(tl)
    out["sparc_tilt_rate"] = sparc(np.abs(d["tilt_rate"].values[mv]))

    # --- lead-lag --------------------------------------------------------------
    # velocities, not positions: positions are ramps, and the shared trend
    # biases a position cross-correlation towards zero lag
    out["xcorr_lag_vel"], out["xcorr_r_vel"] = xcorr_lead_lag(v1m, v2m)
    # vertical (balance) channel: one player moves their end up/down and the
    # partner follows to keep the beam level -> same-sign vertical velocity
    out["xcorr_lag_vy"], out["xcorr_r_vy"] = xcorr_lead_lag(
        d["vy1"].values[mv], d["vy2"].values[mv])
    frac, sw, _ = windowed_lead_lag(v1m, v2m)
    out["frac_time_P1_leads"], out["n_lead_switches"] = frac, sw

    # --- forces: logged ------------------------------------------------------
    if np.isfinite(d["F1"]).all() and np.isfinite(d["F2"]).all():
        f1, f2 = np.abs(d["F1"].values[mv]), np.abs(d["F2"].values[mv])
        out["mean_abs_F1"], out["mean_abs_F2"] = f1.mean(), f2.mean()
        den = f1.mean() + f2.mean()
        out["force_asym_idx"] = (f1.mean() - f2.mean()) / den if den > 0 else np.nan
    if np.isfinite(d["Fh1"]).all():
        out["mean_abs_haptic"] = float(np.mean(np.abs(d["Fh1"].values[mv])))
        out["energy_leader_idx_haptic"] = energy_leader_index(
            d["Fh1"].values[mv], v1m, v2m, dt)

    # --- forces: kinematic -----------------------------------------------------
    # coupling as a pure spring along x -> force on P1 is -(x1-x2) (K cancels)
    out["energy_leader_idx_kin"] = energy_leader_index(-dx, v1m, v2m, dt)
    sub = d.loc[mv]
    out.update(corrective_tilt_metrics(sub))
    return out, (t_move, t_stop)


NORM_SIGNALS = {
    # key in record : column in segment
    "x1": "x1_f", "x2": "x2_f", "v1": "v1", "v2": "v2", "dx": "dx", "dv": "dv",
    "tilt": "tilt", "obj_v": "obj_v", "F1": "F1", "F2": "F2", "Fh1": "Fh1",
    "Fk1": "a1", "Fk2": "a2", "vy1": "vy1", "vy2": "vy2",
}


def normalized_record(d, t_start, t_stop):
    rec = {}
    t = d["t"].values
    for key, col in NORM_SIGNALS.items():
        y = d[col].values
        if key in ("Fk1", "Fk2"):
            y = C.M_EFF_KG * y
        rec[key] = resample_normalized(t, y, t_start, t_stop)
    return rec


# ======================================================================
# PHASE-LEVEL METRICS (need several trials)
# ======================================================================
def curve_variability(curves):
    """Mean over the normalised grid of the across-trial SD."""
    arr = np.asarray(curves, float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.nan
    return float(np.nanmean(np.nanstd(arr, axis=0, ddof=1)))


def linear_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))
