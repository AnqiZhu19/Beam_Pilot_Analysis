"""
Kinematic metrics for the dyadic beam-balancing task.

Sign conventions (all "lead" metrics):  > 0  means  Player 1 (Primary) leads.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert

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
def xcorr_curve(a, b, fs=C.FS, max_lag_s=C.XCORR_MAX_LAG_S):
    """Lead-lag cross-correlation function r(tau), tau in s.
    r(tau) = corr(a(t), b(t + tau)):  a peak at tau > 0 means b repeats a
    tau seconds later, i.e. `a` LEADS by tau.  Returns (lags_s, r)."""
    a = np.asarray(a, float) - np.nanmean(a)
    b = np.asarray(b, float) - np.nanmean(b)
    n = len(a)
    L = int(round(max_lag_s * fs))
    if n < 2 * L + 10:
        L = max(1, n // 4)
    lags = np.arange(-L, L + 1)
    r = np.full(len(lags), np.nan)
    for i, k in enumerate(lags):
        x, y = (a[: n - k], b[k:]) if k >= 0 else (a[-k:], b[: n + k])
        if len(x) >= 5 and x.std() > 0 and y.std() > 0:
            r[i] = np.corrcoef(x, y)[0, 1]
    return lags / fs, r


def xcorr_lead_lag(a, b, fs=C.FS, max_lag_s=C.XCORR_MAX_LAG_S):
    """Peak of the cross-correlation function. Returns (lag_s, r_peak);
    lag > 0 <=> `a` leads. Sub-sample precision by parabolic interpolation."""
    lags, r = xcorr_curve(a, b, fs, max_lag_s)
    if not np.isfinite(r).any():
        return np.nan, np.nan
    i = int(np.nanargmax(r))
    lag = lags[i]
    if 0 < i < len(r) - 1 and np.isfinite(r[i - 1]) and np.isfinite(r[i + 1]):
        den = r[i - 1] - 2 * r[i] + r[i + 1]
        if den != 0:
            lag += 0.5 * (r[i - 1] - r[i + 1]) / den / fs
    return float(lag), float(r[i])


XC_LAGS = np.arange(-C.XCORR_MAX_LAG_S, C.XCORR_MAX_LAG_S + 1e-9, 1.0 / C.XCORR_STORE_FS)


def xcorr_on_grid(a, b):
    """Cross-correlation function resampled on the common XC_LAGS grid."""
    lags, r = xcorr_curve(a, b)
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return np.full(len(XC_LAGS), np.nan)
    return np.interp(XC_LAGS, lags[ok], r[ok], left=np.nan, right=np.nan)


def tlcc_matrix(a, b, n_win=C.TLCC_N_WINDOWS, max_lag_s=0.2, win_s=C.WIN_LEADLAG_S, fs=C.FS):
    """Windowed (time-lagged) cross-correlation along the movement:
    n_win overlapping windows of win_s seconds, centred at equal steps of
    the movement (0-100 %); r(tau) in each. Returns (n_lags, n_win) on
    TLCC_LAGS (|tau| <= max_lag_s)."""
    grid = XC_LAGS[np.abs(XC_LAGS) <= max_lag_s + 1e-9]
    out = np.full((len(grid), n_win), np.nan)
    n = len(a)
    half = max(int(win_s * fs / 2), int(max_lag_s * fs) + 10)
    for w in range(n_win):
        c = int((w + 0.5) / n_win * n)
        s0, s1 = max(0, c - half), min(n, c + half)
        if s1 - s0 < 2 * int(max_lag_s * fs) + 10:
            continue
        lags, r = xcorr_curve(a[s0:s1], b[s0:s1], max_lag_s=max_lag_s)
        ok = np.isfinite(r)
        if ok.sum() >= 3:
            out[:, w] = np.interp(grid, lags[ok], r[ok], left=np.nan, right=np.nan)
    return out


TLCC_LAGS = XC_LAGS[np.abs(XC_LAGS) <= 0.2 + 1e-9]


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


def n_tilt_corrections(tilt, min_amp_deg=1.0):
    """Number of tilt reversals (local extrema of tilt with >= min_amp
    excursion) = discrete balance corrections."""
    tilt = np.asarray(tilt, float)
    pk_hi, _ = find_peaks(tilt, prominence=min_amp_deg)
    pk_lo, _ = find_peaks(-tilt, prominence=min_amp_deg)
    return int(len(pk_hi) + len(pk_lo))


# ======================================================================
# PARTICIPANT FORCE FROM ROBOT DATA + INTERACTION FORCE
# ======================================================================
def corrective_force(f, window_s=C.CORRECTIVE_SMA_WINDOW_S, fs=C.FS):
    """Eq. 7: F_nc = (1/N) sum_t | F(t) - SMA_k(F)(t) |
    SMA = trailing simple moving average over the last k samples
    (k = window_s * fs; at the start of the trial the window is shortened).
    Measures the fast force fluctuations around the slow force trend, i.e.
    the small corrective adjustments. (The printed equation sums j = 0..k but
    divides by k; a true average over the window is used here.)"""
    f = np.asarray(f, float)
    if len(f) < 3 or not np.isfinite(f).any():
        return np.nan, np.full(len(f), np.nan)
    k = max(1, int(round(window_s * fs)))
    sma = pd.Series(f).rolling(k, min_periods=1).mean().values
    dev = f - sma
    return float(np.nanmean(np.abs(dev))), dev


def groten_interaction(f1, f2):
    """Groten et al. 2009 split of two forces on a shared object into the part
    that cancels (internal) and the effective parts (move the object).
    Used here only for the dominance index."""
    f1, f2 = np.asarray(f1, float), np.asarray(f2, float)
    opp = (np.sign(f1) * np.sign(f2)) < 0
    fI = np.where(opp, np.where(np.abs(f1) <= np.abs(f2), f1, -f2), 0.0)
    return fI, f1 - fI, f2 + fI


AXIS_COLS = {"x": dict(p1="x1_f", p2="x2_f", v1="v1", v2="v2", a1="a1", a2="a2"),
             "y": dict(p1="y1_f", p2="y2_f", v1="vy1", v2="vy2", a1="ay1", a2="ay2")}


def force_metrics(d, mv, gains):
    """Viscous-field force model (see config.py). Per axis:
      F_visc_i = B_visc * v_i                      effort against the viscous field
      F_int    = K_c (x1 - x2) + B_c (v1 - v2)     interaction force P1 -> P2
                                                   (P2 transmits -F_int to P1)
      F_h1 = F_visc_1 + F_int (+ M a_1),  F_h2 = F_visc_2 - F_int (+ M a_2)
    All signed (+ = towards +axis). Adds the time series to `d`."""
    out = {}
    dt = 1.0 / C.FS
    obj_dir = np.sign(d["obj_v"].values)                  # direction the beam moves
    for ax, c in AXIS_COLS.items():
        g = (gains or {}).get(ax, {})
        Bv, K, Bc = g.get("B_visc", np.nan), g.get("K_c", np.nan), g.get("B_c", np.nan)
        v1, v2 = d[c["v1"]].values, d[c["v2"]].values
        have_v, have_c = np.isfinite(Bv), np.isfinite(K)
        if not (have_v or have_c):
            continue
        fv1 = Bv * v1 if have_v else np.zeros(len(v1))
        fv2 = Bv * v2 if have_v else np.zeros(len(v2))
        fint = (K * (d[c["p1"]].values - d[c["p2"]].values) + Bc * (v1 - v2)) if have_c else np.zeros(len(v1))
        h1 = fv1 + fint + C.ROBOT_MASS_KG * d[c["a1"]].values
        h2 = fv2 - fint + C.ROBOT_MASS_KG * d[c["a2"]].values
        d[f"Fh1{ax}"], d[f"Fh2{ax}"] = h1, h2
        if have_v:
            d[f"Fv1{ax}"], d[f"Fv2{ax}"] = fv1, fv2
        m1, m2 = h1[mv], h2[mv]
        out[f"mean_Fh_{ax}_P1"], out[f"mean_Fh_{ax}_P2"] = float(m1.mean()), float(m2.mean())
        out[f"peak_Fh_{ax}_P1"] = float(m1[np.argmax(np.abs(m1))])
        out[f"peak_Fh_{ax}_P2"] = float(m2[np.argmax(np.abs(m2))])
        out[f"work_{ax}_P1"] = float(np.sum(m1 * v1[mv]) * dt)
        out[f"work_{ax}_P2"] = float(np.sum(m2 * v2[mv]) * dt)
        if have_v:
            out[f"mean_Fvisc_{ax}_P1"] = float(fv1[mv].mean())
            out[f"mean_Fvisc_{ax}_P2"] = float(fv2[mv].mean())
        if have_c:
            d[f"Fint{ax}"] = fint
            fi = fint[mv]
            out[f"mean_Fint_{ax}"] = float(fi.mean())
            out[f"mean_abs_Fint_{ax}"] = float(np.abs(fi).mean())
            out[f"peak_Fint_{ax}"] = float(fi[np.argmax(np.abs(fi))])
            out[f"pct_interaction_{ax}"] = 100.0 * float(np.mean(np.abs(fi) > C.INTERACTION_DEADBAND_N))
            if ax == "x":
                # interaction along the direction of motion: > 0 P1 pulls/pushes
                # P2 along (P1 leads), < 0 P1 is held back / dragged (P1 follows)
                fl = fi * obj_dir[mv]
                out["Fint_lead_x"] = float(fl.mean())
                act = np.abs(fi) > C.INTERACTION_DEADBAND_N
                out["frac_P1_pulls_x"] = float(np.mean(fl[act] > 0)) if act.any() else np.nan
        # dominance (Groten): share of the effective (object-moving) force
        _, e1, e2 = groten_interaction(m1, m2)
        s1, s2 = np.abs(e1).sum(), np.abs(e2).sum()
        out[f"dominance_P1_{ax}"] = s1 / (s1 + s2) if (s1 + s2) > 0 else np.nan
        # --- corrective forces, Eq. 7 ---------------------------------------
        nc1, dev1 = corrective_force(m1)
        nc2, dev2 = corrective_force(m2)
        out[f"Fnc_{ax}_P1"], out[f"Fnc_{ax}_P2"] = nc1, nc2
        out[f"Fnc_share_{ax}"] = (nc1 - nc2) / (nc1 + nc2) if (nc1 + nc2) > 0 else np.nan
        if have_c:
            out[f"Fnc_int_{ax}"], _ = corrective_force(fint[mv])
        dd1 = np.full(len(d), np.nan); dd1[mv] = dev1
        dd2 = np.full(len(d), np.nan); dd2[mv] = dev2
        d[f"Fnc1{ax}"], d[f"Fnc2{ax}"] = dd1, dd2

    # logged robot forces F1/F2 (commanded, signed) - kept for reference
    if np.isfinite(d["F1"]).all() and np.isfinite(d["F2"]).all():
        f1, f2 = d["F1"].values[mv], d["F2"].values[mv]
        if "Fnc_y_P1" not in out:
            # no vertical gains: Eq. 7 on the logged vertical handle forces
            nc1, dev1 = corrective_force(f1)
            nc2, dev2 = corrective_force(f2)
            out["Fnc_y_P1"], out["Fnc_y_P2"] = nc1, nc2
            out["Fnc_share_y"] = (nc1 - nc2) / (nc1 + nc2) if (nc1 + nc2) > 0 else np.nan
            out["Fnc_y_from_logged_F"] = 1.0
            dd1 = np.full(len(d), np.nan); dd1[mv] = dev1
            dd2 = np.full(len(d), np.nan); dd2[mv] = dev2
            d["Fnc1y"], d["Fnc2y"] = dd1, dd2
        out["mean_F1"], out["mean_F2"] = float(f1.mean()), float(f2.mean())
        a1, a2 = np.abs(f1).mean(), np.abs(f2).mean()
        out["load_share_idx"] = (a1 - a2) / (a1 + a2) if (a1 + a2) > 0 else np.nan

    # validity of neglecting the mass term: M must be << B v / a
    for ax, c in AXIS_COLS.items():
        Bv = (gains or {}).get(ax, {}).get("B_visc", np.nan)
        if np.isfinite(Bv):
            v = np.abs(d[c["v1"]].values[mv]); a = np.abs(d[c["a1"]].values[mv])
            out[f"mass_limit_kg_{ax}"] = float(Bv * np.percentile(v, 90) / max(np.percentile(a, 90), 1e-9))
    return out


# ======================================================================
# RELATIVE PHASE
# ======================================================================
def phase_plane_angle(x, v):
    """Movement-progress phase (deg) from the normalised position-velocity
    phase plane: 0 deg at the start, 90 at peak velocity, 180 at the end.
    Works for discrete (non-oscillatory) point-to-point movements."""
    x, v = np.asarray(x, float), np.asarray(v, float)
    direction = np.sign(np.nansum(v)) or 1.0
    x, v = direction * x, direction * v
    rng = np.nanmax(x) - np.nanmin(x)
    vmax = np.nanmax(np.abs(v))
    if rng <= 0 or vmax <= 0:
        return np.full(len(x), np.nan)
    xn = 2 * (x - np.nanmin(x)) / rng - 1
    vn = v / vmax
    return np.degrees(np.unwrap(np.arctan2(vn, -xn)))


def continuous_relative_phase(x1, v1, x2, v2):
    """CRP = phase1 - phase2 (deg). > 0: P1 is further through the movement
    cycle = P1 leads. Each player is normalised with their own amplitude."""
    return phase_plane_angle(x1, v1) - phase_plane_angle(x2, v2)


def hilbert_relative_phase(a, b):
    """Relative phase of two oscillatory signals via the Hilbert transform.
    Returns (mean relative phase deg, phase-locking value 0..1).
    Mean > 0: `a` leads; PLV = 1 perfectly locked, 0 = no consistent relation."""
    a = np.asarray(a, float) - np.nanmean(a)
    b = np.asarray(b, float) - np.nanmean(b)
    if len(a) < 20 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    rel = np.angle(hilbert(a)) - np.angle(hilbert(b))
    z = np.mean(np.exp(1j * rel))
    return float(np.degrees(np.angle(z))), float(np.abs(z))


def circ_mean_sd_deg(phi):
    """Circular mean and circular SD (deg) of an angle series in degrees."""
    phi = np.radians(np.asarray(phi, float))
    phi = phi[np.isfinite(phi)]
    if len(phi) == 0:
        return np.nan, np.nan
    z = np.mean(np.exp(1j * phi))
    R = min(np.abs(z), 1.0)
    sd = np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else np.nan
    return float(np.degrees(np.angle(z))), float(sd)


# ======================================================================
# PER-TRIAL METRICS
# ======================================================================
def trial_metrics(d, gains=None):
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
    mt = out["movement_time"]
    out["xcorr_lag_vel_pct"] = 100 * out["xcorr_lag_vel"] / mt if mt and np.isfinite(mt) else np.nan
    d.attrs["xc_vel"] = xcorr_on_grid(v1m, v2m)
    d.attrs["xc_vy"] = xcorr_on_grid(d["vy1"].values[mv], d["vy2"].values[mv])
    d.attrs["tlcc_vel"] = tlcc_matrix(v1m, v2m)
    frac, sw, _ = windowed_lead_lag(v1m, v2m)
    out["frac_time_P1_leads"], out["n_lead_switches"] = frac, sw

    # --- forces: signed, logged + participant + interaction ------------------
    out.update(force_metrics(d, mv, gains))
    if np.isfinite(d["Fh1"]).all():
        out["mean_haptic"] = float(np.mean(d["Fh1"].values[mv]))
        out["mean_abs_haptic"] = float(np.mean(np.abs(d["Fh1"].values[mv])))
        out["energy_leader_idx_haptic"] = energy_leader_index(
            d["Fh1"].values[mv], v1m, v2m, dt)

    # --- relative phase ------------------------------------------------------
    # (a) movement axis, discrete reach: phase-plane continuous relative phase
    crp = continuous_relative_phase(d["x1_f"].values[mv], v1m, d["x2_f"].values[mv], v2m)
    d["crp"] = np.nan
    d.loc[mv, "crp"] = crp
    sp_ok = (np.abs(v1m) > thr) | (np.abs(v2m) > thr)    # phase undefined at rest
    out["crp_mean_deg"], out["crp_sd_deg"] = circ_mean_sd_deg(crp[sp_ok])
    # (b) vertical balance channel, oscillatory: Hilbert phase + phase locking
    out["relphase_vy_deg"], out["plv_vy"] = hilbert_relative_phase(
        d["vy1"].values[mv], d["vy2"].values[mv])
    # (c) Hilbert relative phase of the horizontal velocities (whole movement)
    out["relphase_vel_deg"], out["plv_vel"] = hilbert_relative_phase(v1m, v2m)

    # --- forces: kinematic -----------------------------------------------------
    # coupling as a pure spring along x -> force on P1 is -(x1-x2) (K cancels)
    out["energy_leader_idx_kin"] = energy_leader_index(-dx, v1m, v2m, dt)
    return out, (t_move, t_stop)


NORM_SIGNALS = {
    # key in record : column in segment
    "x1": "x1_f", "x2": "x2_f", "v1": "v1", "v2": "v2", "dx": "dx", "dv": "dv",
    "tilt": "tilt", "obj_v": "obj_v", "vy1": "vy1", "vy2": "vy2", "crp": "crp",
    "F1": "F1", "F2": "F2", "haptic": "Fh1",
    "Fh1x": "Fh1x", "Fh2x": "Fh2x", "Fh1y": "Fh1y", "Fh2y": "Fh2y",
    "Fv1x": "Fv1x", "Fv2x": "Fv2x", "Fintx": "Fintx", "Finty": "Finty",
    "Fnc1x": "Fnc1x", "Fnc2x": "Fnc2x", "Fnc1y": "Fnc1y", "Fnc2y": "Fnc2y",
}
REALTIME_SIGNALS = ["x1", "x2", "v1", "v2", "dx", "tilt"]
RT_GRID = np.arange(0, C.REALTIME_MAX_S, 1.0 / C.REALTIME_FS)


def normalized_record(d, t_start, t_stop):
    """Per-trial curves: 0-100 % movement-normalised (key) and real time from
    movement onset (key 'rt_<name>', NaN after the trial ended)."""
    rec = {}
    t = d["t"].values
    for key, col in NORM_SIGNALS.items():
        if col not in d:
            rec[key] = np.full(C.N_GRID, np.nan)
            continue
        y = d[col].values.astype(float)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            rec[key] = np.full(C.N_GRID, np.nan)
            continue
        rec[key] = resample_normalized(t[ok], y[ok], t_start, t_stop)
    for key in ("xc_vel", "xc_vy", "tlcc_vel"):
        if key in d.attrs:
            rec[key] = d.attrs[key]
    if np.isfinite(t_start):
        tr = t_start + RT_GRID
        for key in REALTIME_SIGNALS:
            col = NORM_SIGNALS[key]
            y = np.interp(tr, t, d[col].values.astype(float))
            y[tr > t_stop] = np.nan
            rec["rt_" + key] = y
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
