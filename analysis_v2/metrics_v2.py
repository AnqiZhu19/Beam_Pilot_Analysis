"""
Per-trial metrics. Sign convention: every lead/lag/difference metric is > 0
when P1 (Primary) leads, initiates first, finishes first or is ahead.
"""
import sys

import numpy as np

import config_v2 as C
from io_v2 import onset_offset
import segmentation as S

sys.path.insert(0, C.OLD_CODE_DIR)
import metrics as OM  # noqa: E402  (old pilot_analysis metrics: sparc, xcorr, tlcc)

PHASES = ["acc", "dec", "final"]
PHASE_LONG = {"acc": "acceleration\n(onset -> peak v)", "dec": "deceleration\n(peak v -> 1st in target)",
              "final": "final approach\n(1st in target -> end)"}
INTER_CODE = {k: i for i, k in enumerate(S.INTERACTION)}
NORM_KEYS = ["x1", "x2", "dx", "y1", "y2", "theta", "vx1", "vx2", "obj_v"]
RT_T = np.arange(0, C.RT_MAX_S + 1e-9, 1.0 / C.RT_FS)
TLCC_LAGS = OM.TLCC_LAGS


# ----------------------------------------------------------------------
def corrective_movement(z, fs=C.FS, win_s=C.CORRECTIVE_WIN_S):
    """Paper Eq. 11: mean |z(t) - mean(z(t-j), j = 0..min(k,t)-1)|."""
    z = np.asarray(z, float)
    k = max(1, int(round(win_s * fs)))
    cs = np.r_[0.0, np.cumsum(z)]
    idx = np.arange(1, len(z) + 1)
    lo = np.maximum(idx - k, 0)
    ma = (cs[idx] - cs[lo]) / (idx - lo)
    return float(np.mean(np.abs(z - ma)))


def _ratio(a, b):
    return float(a / (a + b)) if np.isfinite(a + b) and a + b > 0 else np.nan


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _xcorr_abs_peak(a, b, max_lag_s=0.5):
    """Lag (s) and r at the peak of |r(tau)|; lag > 0 => a leads b."""
    lags, r = OM.xcorr_curve(a, b, C.FS, max_lag_s)
    if not np.isfinite(r).any():
        return np.nan, np.nan
    i = int(np.nanargmax(np.abs(r)))
    if i in (0, len(r) - 1):          # peak at the edge of the search range: no lag found
        return np.nan, float(r[i])
    return float(lags[i]), float(r[i])


def _gated_lag(a, b):
    """xcorr lead-lag (> 0: P1 leads); lag set to NaN if the peak r < XCORR_MIN_R."""
    lag, r = OM.xcorr_lead_lag(a, b, C.FS, C.XCORR_MAX_LAG_S)
    ok = np.isfinite(r) and r >= C.XCORR_MIN_R and abs(lag) < C.XCORR_MAX_LAG_S - 1.5 / C.FS  # no edge peaks
    return (lag if ok else np.nan), r


def main_reach_segment(lab, t, x):
    """Forward HMM segment with the largest displacement."""
    best = None
    for a, b, v in S.runs(lab):
        amp = x[b - 1] - x[a]
        if v == 1 and amp >= C.MIN_SEG_AMP_X and (best is None or amp > best[2]):
            best = (t[a], t[b - 1], amp)
    return best if best else (np.nan, np.nan, np.nan)


def paired_segment_delays(lab1, lab2, t, x1, x2, max_sep_s=0.5):
    """'Delay in our way': every non-stationary X segment of P1 (|amp| >= MIN_SEG_AMP_X)
    is paired with the P2 segment of the same direction whose onset is closest
    (segments must overlap in time, onsets within max_sep_s). delay = onset_P2 - onset_P1 (> 0: P1 initiated first)."""
    def segs(lab, x):
        return [(t[a], t[b - 1], v) for a, b, v in S.runs(lab)
                if v != 0 and abs(x[b - 1] - x[a]) >= C.MIN_SEG_AMP_X]
    s1, s2 = segs(lab1, x1), segs(lab2, x2)
    out = []
    for on1, off1, v in s1:
        # same direction, overlapping in time, onsets within max_sep_s
        cand = [(abs(on2 - on1), on2, off2) for on2, off2, w in s2
                if w == v and abs(on2 - on1) <= max_sep_s and on2 < off1 and on1 < off2]
        if cand:
            _, on2, off2 = min(cand)
            out.append(dict(direction=int(v), onset_delay=on2 - on1, offset_delay=off2 - off1))
    return out


# ----------------------------------------------------------------------
def trial_metrics(d, row, lab):
    """d: resampled trial; row: trial-table row; lab: dict of HMM label arrays
    (v1, v2 vertical; beam; r1, r2 reach). Returns (metrics dict, curves dict, segment rows)."""
    t = d["t"].values
    m, cv = {}, {}
    x1, x2, y1, y2 = (d[k].values for k in ("x1", "x2", "y1", "y2"))
    vx1, vx2, obj_v = d["vx1"].values, d["vx2"].values, d["obj_v"].values
    th, thd = d["theta"].values, d["theta_dot"].values

    # ---------- movement window (shared object) and per-hand onsets ----------
    t_on, t_off = onset_offset(np.abs(obj_v), t)
    pk = max(np.nanmax(np.abs(vx1)), np.nanmax(np.abs(vx2)))   # shared threshold
    on1, off1 = onset_offset(np.abs(vx1), t, pk)
    on2, off2 = onset_offset(np.abs(vx2), t, pk)
    if not np.isfinite(t_on):
        t_on, t_off = t[0], t[-1]
    mv = (t >= t_on) & (t <= t_off)
    ipk = np.argmax(np.where(mv, obj_v, -np.inf))
    t_pk = t[ipk]
    m.update(dict(t_onset=t_on, t_offset=t_off, movement_time=t_off - t_on,
                  trial_time=t[-1], t_peak_obj_v=t_pk, peak_obj_v=float(obj_v[ipk]),
                  onset_P1=on1, onset_P2=on2, offset_P1=off1, offset_P2=off2))

    # ---------- 1. TRAJECTORY ----------
    dxm = d["dx"].values[mv]
    m["mean_dx_mm"] = 1e3 * float(np.mean(dxm))
    m["mean_abs_dx_mm"] = 1e3 * float(np.mean(np.abs(dxm)))
    m["rms_dx_mm"] = 1e3 * float(np.sqrt(np.mean(dxm ** 2)))
    m["peak_abs_dx_mm"] = 1e3 * float(np.max(np.abs(dxm)))
    m["mean_abs_dy_mm"] = 1e3 * float(np.mean(np.abs(d["dy"].values)))
    m["end_x_P1"], m["end_x_P2"] = float(x1[-1]), float(x2[-1])
    m["end_dx_mm"] = 1e3 * float(x1[-1] - x2[-1])
    # corrective movement (Eq. 11) on the vertical axis (paper) and the reach axis
    for ax, z1, z2 in (("y", y1, y2), ("x", x1, x2)):
        c1, c2 = corrective_movement(z1), corrective_movement(z2)
        m[f"corr_{ax}_P1_mm"], m[f"corr_{ax}_P2_mm"] = 1e3 * c1, 1e3 * c2
        m[f"CR_{ax}_P1"] = _ratio(c1, c2)
    # smoothness on the movement window
    sp1 = np.hypot(vx1, d["vy1"].values)[mv]
    sp2 = np.hypot(vx2, d["vy2"].values)[mv]
    m["sparc_P1"], m["sparc_P2"] = OM.sparc(sp1, C.FS), OM.sparc(sp2, C.FS)
    m["sparc_obj"] = OM.sparc(np.abs(obj_v[mv]), C.FS)
    m["n_submov_P1"], m["n_submov_P2"] = OM.n_submovements(sp1, C.FS), OM.n_submovements(sp2, C.FS)

    # ---------- 2. BEAM TILT ----------
    m["max_abs_theta"] = float(np.max(np.abs(th)))
    m["rms_theta"] = float(np.sqrt(np.mean(th ** 2)))
    m["mean_theta"] = float(np.mean(th))
    m["sd_theta"] = float(np.std(th))
    m["end_theta"] = float(th[-1])
    m["r_theta_thetadot"] = _pearson(th, thd)
    m["r_theta_objv"] = _pearson(th, obj_v)
    m["r_absthetadot_speed"] = _pearson(np.abs(thd), np.abs(obj_v))
    m["lag_objv_theta"], m["r_lag_objv_theta"] = _xcorr_abs_peak(obj_v, th)   # > 0: velocity leads tilt
    m["r_theta_dx"] = _pearson(th, d["dx"].values)
    # beam rotation segments (HMM on theta_dot)
    rot = [r for r in S.runs(lab["beam"]) if r[2] != 0]
    m["n_rot_segments"] = len(rot)
    m["mean_rot_dur"] = float(np.mean([(b - a) / C.FS for a, b, _ in rot])) if rot else np.nan
    m["frac_time_rotating"] = float(np.mean(lab["beam"] != 0))
    im, inter = S.interaction_metrics(lab["v1"], lab["v2"], y1, y2)
    m.update(im)

    # ---------- 3. TEMPORAL (X reach) ----------
    m["init_diff"] = on2 - on1                      # > 0 P1 starts first
    m["complete_diff"] = off2 - off1                # > 0 P1 stops first
    m["target_entry_diff"] = row.get("target_entry_diff", np.nan)
    m["P1_initiates"] = float(m["init_diff"] > 0) if np.isfinite(m["init_diff"]) else np.nan
    m["P1_completes"] = float(m["complete_diff"] > 0) if np.isfinite(m["complete_diff"]) else np.nan
    m["P1_enters_target"] = (float(m["target_entry_diff"] > 0)
                             if np.isfinite(m["target_entry_diff"]) else np.nan)
    pk1, pk2 = t[np.argmax(np.where(mv, vx1, -np.inf))], t[np.argmax(np.where(mv, vx2, -np.inf))]
    m["peak_vel_lag"] = pk2 - pk1
    # HMM main reach segment
    a1, b1, _ = main_reach_segment(lab["r1"], t, x1)
    a2, b2, _ = main_reach_segment(lab["r2"], t, x2)
    m["hmm_init_diff"], m["hmm_complete_diff"] = a2 - a1, b2 - b1
    # paired X-segment delays
    pairs = paired_segment_delays(lab["r1"], lab["r2"], t, x1, x2)
    for name, sel in (("all", lambda p: True), ("fwd", lambda p: p["direction"] == 1),
                      ("bwd", lambda p: p["direction"] == -1)):
        dl = np.array([p["onset_delay"] for p in pairs if sel(p)])
        m[f"seg_delay_{name}"] = float(dl.mean()) if len(dl) else np.nan
        m[f"seg_absdelay_{name}"] = float(np.abs(dl).mean()) if len(dl) else np.nan
        m[f"n_seg_pairs_{name}"] = len(dl)
    dl = np.array([p["onset_delay"] for p in pairs])
    m["seg_frac_P1_first"] = float(np.mean(dl > 0)) if len(dl) else np.nan
    for i, (lb, x) in enumerate(((lab["r1"], x1), (lab["r2"], x2)), 1):
        sg = [(a, b, v) for a, b, v in S.runs(lb) if v != 0 and abs(x[b - 1] - x[a]) >= C.MIN_SEG_AMP_X]
        m[f"n_reach_seg_P{i}"] = len(sg)
        m[f"n_backward_P{i}"] = sum(1 for *_, v in sg if v == -1)
    # whole-movement xcorr on X velocity
    m["xcorr_lag_vx"], m["xcorr_r_vx"] = _gated_lag(vx1[mv], vx2[mv])
    # movement phases: acceleration / deceleration / final approach
    te = np.nanmin([row.get("target_entry_P1", np.nan), row.get("target_entry_P2", np.nan)])
    te = float(np.clip(te if np.isfinite(te) else t_off, t_pk, t[-1]))
    m["t_first_in_target"] = te
    bounds = {"acc": (t_on, t_pk), "dec": (t_pk, te), "final": (te, t[-1])}
    for ph, (a, b) in bounds.items():
        sel = (t >= a) & (t <= b)
        m[f"{ph}_dur"] = float(b - a)
        if sel.sum() < 5:
            for k in ("mean_dx_mm", "frac_P1_ahead", "lag_vx", "r_vx", "aligned", "corr_y_P1_mm", "corr_y_P2_mm"):
                m[f"{ph}_{k}"] = np.nan
            continue
        m[f"{ph}_mean_dx_mm"] = 1e3 * float(np.mean(d["dx"].values[sel]))
        m[f"{ph}_frac_P1_ahead"] = float(np.mean(d["dx"].values[sel] > 0))
        if b - a >= C.XCORR_PHASE_MIN_S:
            m[f"{ph}_lag_vx"], m[f"{ph}_r_vx"] = _gated_lag(vx1[sel], vx2[sel])
        else:
            m[f"{ph}_lag_vx"], m[f"{ph}_r_vx"] = np.nan, np.nan
        m[f"{ph}_aligned"] = float(np.mean(inter[sel] == "aligned"))
        m[f"{ph}_corr_y_P1_mm"] = 1e3 * corrective_movement(y1[sel])
        m[f"{ph}_corr_y_P2_mm"] = 1e3 * corrective_movement(y2[sel])

    # ---------- curves ----------
    g = np.linspace(t_on, t_off, C.N_GRID)
    for k in NORM_KEYS:
        cv["n_" + k] = np.interp(g, t, d[k].values)
    tr = t_on + RT_T
    inside = tr <= t[-1]
    for k in NORM_KEYS + ["theta_dot"]:
        cv["rt_" + k] = np.where(inside, np.interp(tr, t, d[k].values), np.nan)
    codes = np.array([INTER_CODE[s] for s in inter], float)
    cv["rt_inter"] = np.where(inside, codes[np.clip(np.searchsorted(t, tr), 0, len(t) - 1)], np.nan)
    cv["tlcc"] = OM.tlcc_matrix(vx1[mv], vx2[mv]) if mv.sum() > 0.6 * C.FS else None

    # ---------- segment table ----------
    segs = []
    for key, kind, who, pos in (("v1", "vert", "P1", y1), ("v2", "vert", "P2", y2),
                                ("beam", "beam", "beam", th),
                                ("r1", "reach", "P1", x1), ("r2", "reach", "P2", x2)):
        segs += S.segment_table(lab[key], t, pos, kind, who)
    return m, cv, segs, inter
