"""
Per-dyad analysis.  One output folder per pilot:

    Results/<id>/
        trial_summary.csv      one row per trial, all metrics
        phase_summary.csv      one row per phase (means, variability, learning slopes)
        early_late_summary.csv first vs last N trials of every phase
        qc_report.txt          clock sync, data gaps, sanity checks
        figures/fig00 ... fig10

Run all dyads:        python3 run_analysis.py
Run selected dyads:   python3 run_analysis.py Pilot2 Pilot3
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings as _w
import numpy as np
import pandas as pd

# all-NaN slices (e.g. a metric undefined in one phase) are expected
_w.filterwarnings("ignore", category=RuntimeWarning)

import config as C
import data_utils as U
import metrics as M
import plot_utils as P
from plot_utils import plt

PHASE_METRICS = [m[0] for m in P.KEY_METRICS] + [
    "trial_duration", "peak_speed_P1", "peak_speed_P2", "mean_speed_P1", "mean_speed_P2",
    "peak_speed_lag", "init_time_diff", "target_time_diff", "n_submov_P1", "n_submov_P2",
    "n_tilt_corrections", "sd_tilt", "mean_tilt", "mean_dx", "peak_abs_dx",
    "frac_opposing_motion", "xcorr_r_vel", "xcorr_lag_vy", "n_lead_switches",
    "corr_disp_share_idx", "energy_leader_idx_haptic", "mean_abs_F1", "mean_abs_F2",
    "mean_abs_haptic", "rms_kin_force_P1", "rms_kin_force_P2",
    "traj_dev_P1", "traj_dev_P2",
]


# ======================================================================
# PROCESSING
# ======================================================================
def process_pilot(pcfg, verbose=True):
    """Load, sync, segment and compute every metric for one dyad.
    Returns a dict used both here and by overlay_analysis.py."""
    log = []

    def say(msg):
        log.append(msg)
        if verbose:
            print("  " + msg)

    trial, p, s = U.load_raw(pcfg["folder"])
    tt = U.build_trial_table(trial)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tt, found, ok = U.assign_phases(tt, pcfg["protocol"])
    say(f"protocol {pcfg['protocol']}; phases found: {found} -> "
        f"{'OK' if ok else 'MISMATCH (check config.PILOTS)'}")
    for ph, n in tt.groupby("phase", sort=False).size().items():
        say(f"    {ph:12s}: {n} trials")

    offset, n_match, n_win, spread = U.detect_time_offset(tt, p)
    U.apply_time_offset(p, s, offset)
    say(f"clock sync: offset {offset:.4f} s, {n_match}/{len(tt)} trials matched to "
        f"{n_win} log windows (residual SD {spread:.4f} s)")
    if n_match < 0.9 * len(tt):
        say("  WARNING: <90 % of trials matched a log window - check the sync!")

    segs, recs, rows, qcs = {}, {}, [], []
    for _, row in tt.iterrows():
        key = (row["block"], row["trial"])
        d, qc = U.segment_trial(row, p, s)
        qc.update(block=row["block"], trial=row["trial"])
        qcs.append(qc)
        out = row.to_dict()
        if d is not None:
            mtr, (t0, t1) = M.trial_metrics(d)
            out.update(mtr)
            segs[key] = d
            recs[key] = M.normalized_record(d, t0, t1)
        rows.append(out)
    df = pd.DataFrame(rows)
    qc = pd.DataFrame(qcs)
    say(f"{len(segs)}/{len(tt)} trials segmented")

    # trajectory deviation from this dyad's Baseline-1 mean path
    ref = [recs[k] for k in recs if df.set_index(["block", "trial"]).loc[k, "phase"] == "Baseline 1"]
    for i in (1, 2):
        if ref:
            ref_curve = np.nanmean([r[f"x{i}"] for r in ref], axis=0)
            df[f"traj_dev_P{i}"] = [
                M.rmse(recs[(b, t)][f"x{i}"], ref_curve) if (b, t) in recs else np.nan
                for b, t in zip(df["block"], df["trial"])]
        else:
            df[f"traj_dev_P{i}"] = np.nan

    # QC summary
    if len(qc):
        if "max_gap_p" in qc:
            ng = int(((qc["max_gap_p"] > C.MAX_GAP_S) | (qc["max_gap_s"] > C.MAX_GAP_S)).sum())
            say(f"trials with a data gap > {C.MAX_GAP_S*1000:.0f} ms: {ng}")
        for c, txt in [("x2_rms_diff", "Secondary X: own log vs Primary's OtherPosX, RMS diff"),
                       ("y2_rms_diff", "Secondary Y: own log vs Primary's OtherPosY, RMS diff"),
                       ("relx_rms_vs_x1_minus_x2", "RelativePosX vs (x1 - x2), RMS diff")]:
            if c in qc:
                say(f"{txt}: median {np.nanmedian(qc[c]) * 1000:.2f} mm")
    if "energy_leader_idx_haptic" in df:
        r = df[["energy_leader_idx_kin", "energy_leader_idx_haptic"]].corr().iloc[0, 1]
        say(f"energy-leadership index, kinematic vs logged HapticForce: r = {r:.2f}"
            + ("  (negative -> HapticForce sign is opposite to 'force on P1')" if r < 0 else ""))
    zero_tilt = df["mean_tilt"].median()
    say(f"median signed tilt across trials: {zero_tilt:.2f} deg"
        + ("  <-- large: check y-offset between robots / SECONDARY_Y_SOURCE" if abs(zero_tilt) > 3 else ""))

    phase_df, el_df = phase_summaries(df, recs, pcfg)
    return dict(cfg=pcfg, tt=tt, df=df, segs=segs, recs=recs, qc=qc,
                phase_df=phase_df, el_df=el_df, log=log,
                phases=[ph for ph in C.PROTOCOLS[pcfg["protocol"]]["phases"] if ph in set(df["phase"])])


def _phase_recs(df, recs, phase, subset=None):
    g = df[df["phase"] == phase]
    if subset is not None:
        g = g[g["early_late"] == subset]
    return [recs[(b, t)] for b, t in zip(g["block"], g["trial"]) if (b, t) in recs]


def phase_summaries(df, recs, pcfg):
    rows, el_rows = [], []
    cols = [c for c in PHASE_METRICS if c in df.columns]
    for ph, g in df.groupby("phase", sort=False):
        r = dict(pilot=pcfg["id"], protocol=pcfg["protocol"], phase=ph, n_trials=len(g))
        for c in cols:
            r[c] = g[c].astype(float).mean()
        # variability (across trials, on the normalised curves)
        R = _phase_recs(df, recs, ph)
        for k in ("x1", "x2", "tilt", "v1", "v2", "dx"):
            r[f"var_{k}"] = M.curve_variability([q[k] for q in R])
        r["endpoint_sd_P1"] = np.nanstd([q["x1"][-1] for q in R], ddof=1) if len(R) > 1 else np.nan
        r["endpoint_sd_P2"] = np.nanstd([q["x2"][-1] for q in R], ddof=1) if len(R) > 1 else np.nan
        mt = g["movement_time"].astype(float)
        r["cv_movement_time"] = mt.std() / mt.mean() if mt.mean() > 0 else np.nan
        # learning / stability across trials
        for c in ("max_abs_tilt", "rms_tilt", "movement_time", "mean_abs_dx",
                  "xcorr_lag_vel", "energy_leader_idx_kin", "sparc_object"):
            if c in g:
                r[f"slope_{c}"] = M.linear_slope(g["trial_in_phase"], g[c])
        rows.append(r)

        for part in ("early", "late"):
            gp = g[g["early_late"] == part]
            e = dict(pilot=pcfg["id"], protocol=pcfg["protocol"], phase=ph, part=part, n_trials=len(gp))
            for c in cols:
                e[c] = gp[c].astype(float).mean()
            Rp = _phase_recs(df, recs, ph, part)
            for k in ("x1", "x2", "tilt"):
                e[f"var_{k}"] = M.curve_variability([q[k] for q in Rp])
            el_rows.append(e)
    return pd.DataFrame(rows), pd.DataFrame(el_rows)


# ======================================================================
# FIGURES (one dyad)
# ======================================================================
def _recs_of(res, ph):
    return _phase_recs(res["df"], res["recs"], ph)


def fig00_qc(res, fd):
    df, segs = res["df"], res["segs"]
    if not segs:
        return
    key = list(segs)[len(segs) // 2]
    d = segs[key]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    ax[0, 0].plot(d["t"], d["x1"], color="k", lw=0.6, alpha=0.4, label="x1 raw")
    ax[0, 0].plot(d["t"], d["x1_f"], color="C0", label="x1 filtered (P1)")
    ax[0, 0].plot(d["t"], d["x2_f"], color="C0", ls="--", label="x2 filtered (P2)")
    if np.isfinite(d.attrs.get("t_move", np.nan)):
        ax[0, 0].axvline(d.attrs["t_move"], color="grey", ls=":", label="first home-leave")
    ax[0, 0].set(xlabel="Time in trial (s)", ylabel="x (m)", title=f"Example trial block {key[0]} trial {key[1]}")
    ax[0, 0].legend()
    ax[0, 1].plot(d["t"], d["v1"], label="v1")
    ax[0, 1].plot(d["t"], d["v2"], ls="--", label="v2")
    ax[0, 1].set(xlabel="Time in trial (s)", ylabel="velocity x (m/s)", title="Filtered velocity")
    ax[0, 1].legend()
    ax[1, 0].plot(d["t"], d["tilt"], color="purple")
    ax[1, 0].axhspan(-C.TILT_TOLERANCE_DEG, C.TILT_TOLERANCE_DEG, color="green", alpha=0.08)
    ax[1, 0].set(xlabel="Time in trial (s)", ylabel="tilt (deg)", title="Beam tilt (+ = P1 end higher)")
    ax[1, 1].plot(df["global_trial_idx"], df["trial_duration"], "o-", ms=3, label="trial duration")
    ax[1, 1].plot(df["global_trial_idx"], df["movement_time"], "s-", ms=3, label="movement time")
    P.shade_phases(ax[1, 1], df)
    ax[1, 1].set(xlabel="Trial", ylabel="s", title="Durations (shading = phase)")
    ax[1, 1].legend()
    fig.suptitle(f"{res['cfg']['id']} - QC", fontsize=11)
    P.save(fig, f"{fd}/fig00_qc.png")


def fig01_trajectories(res, fd):
    phases = res["phases"]
    fig, axes = plt.subplots(2, len(phases), figsize=(4 * len(phases), 6.5), sharey="row", squeeze=False)
    for j, ph in enumerate(phases):
        R = _recs_of(res, ph)
        cmap = plt.get_cmap("viridis")
        for k, r in enumerate(R):
            c = cmap(k / max(len(R) - 1, 1))
            axes[0, j].plot(P.PCT, r["x1"], color=c, lw=0.6, alpha=0.6)
            axes[0, j].plot(P.PCT, r["x2"], color=c, lw=0.6, alpha=0.6, ls=C.P2_LS)
            axes[1, j].plot(P.PCT, r["tilt"], color=c, lw=0.6, alpha=0.6)
        if R:
            axes[0, j].plot(P.PCT, np.nanmean([r["x1"] for r in R], 0), "k", lw=2, label="P1 mean")
            axes[0, j].plot(P.PCT, np.nanmean([r["x2"] for r in R], 0), "k", lw=2, ls=C.P2_LS, label="P2 mean")
            axes[1, j].plot(P.PCT, np.nanmean([r["tilt"] for r in R], 0), "k", lw=2)
        axes[0, j].set_title(f"{P.phase_label(ph)}  (n={len(R)})")
        axes[1, j].axhline(0, color="grey", lw=0.7)
        axes[1, j].set_xlabel("Movement completed (%)")
    axes[0, 0].set_ylabel("Hand position x (m)")
    axes[1, 0].set_ylabel("Beam tilt (deg)")
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{res['cfg']['id']} - Fig 1. Trajectories (colour: first trial = purple -> last = yellow;"
                 " P1 solid, P2 dashed)", fontsize=11)
    P.save(fig, f"{fd}/fig01_trajectories.png")


def fig02_velocity(res, fd):
    phases = res["phases"]
    fig, axes = plt.subplots(2, len(phases), figsize=(4 * len(phases), 6), sharey="row", squeeze=False)
    for j, ph in enumerate(phases):
        R = _recs_of(res, ph)
        if not R:
            continue
        c = P.phase_color(ph)
        P.band(axes[0, j], P.PCT, [r["v1"] for r in R], c, C.P1_LS, "P1")
        P.band(axes[0, j], P.PCT, [r["v2"] for r in R], c, C.P2_LS, "P2")
        P.band(axes[1, j], P.PCT, [r["dv"] for r in R], c, "-", "v1 - v2")
        axes[0, j].set_title(f"{P.phase_label(ph)} (n={len(R)})")
        for a in axes[:, j]:
            a.axhline(0, color="grey", lw=0.7)
        axes[1, j].set_xlabel("Movement completed (%)")
    axes[0, 0].set_ylabel("Hand velocity x (m/s)")
    axes[1, 0].set_ylabel("Relative velocity v1 - v2 (m/s)")
    axes[0, 0].legend()
    fig.suptitle(f"{res['cfg']['id']} - Fig 2. Velocity profiles (mean +/- SD across trials)", fontsize=11)
    P.save(fig, f"{fd}/fig02_velocity.png")


def fig03_position_difference(res, fd):
    df, phases = res["df"], res["phases"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ph in phases:
        R = _recs_of(res, ph)
        if R:
            P.band(axes[0], P.PCT, [r["dx"] for r in R], P.phase_color(ph), label=P.phase_label(ph))
    axes[0].axhline(0, color="grey", lw=0.7, ls="--")
    axes[0].set(xlabel="Movement completed (%)", ylabel="x1 - x2 (m)",
                title="(a) Position difference (mean +/- SD)")
    axes[0].legend()
    P.box_by_phase(axes[1], {ph: df.loc[df.phase == ph, "mean_abs_dx"] for ph in phases}, phases,
                   "Mean |x1 - x2| (m)", "(b) Spatial disagreement per trial")
    P.series_by_trial(axes[2], df, "mean_abs_dx", "Mean |x1 - x2| (m)")
    axes[2].set_title("(c) Across trials (line = linear fit per phase)")
    fig.suptitle(f"{res['cfg']['id']} - Fig 3. Hand-position difference", fontsize=11)
    P.save(fig, f"{fd}/fig03_position_difference.png")


def fig04_across_trials(res, fd):
    df = res["df"]
    mets = [m for m in P.KEY_METRICS if m[0] in df.columns]
    nc = 4
    nr = int(np.ceil(len(mets) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3 * nr), squeeze=False)
    for a, (col, lab, zero, note) in zip(axes.ravel(), mets):
        P.series_by_trial(a, df, col, lab, zero=zero, label_phases=(a is axes[0, 0]))
        if note:
            a.text(0.01, 0.99, note, transform=a.transAxes, va="top", fontsize=7, color="grey")
    axes[0, 0].legend(fontsize=6.5)
    for a in axes.ravel()[len(mets):]:
        a.axis("off")
    fig.suptitle(f"{res['cfg']['id']} - Fig 4. Evolution across trials (shading/colour = phase, "
                 "thick line = linear trend within phase)", fontsize=11)
    P.save(fig, f"{fd}/fig04_across_trials.png")


def fig05_beam_tilt(res, fd):
    df, phases = res["df"], res["phases"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ph in phases:
        R = _recs_of(res, ph)
        if R:
            P.band(axes[0, 0], P.PCT, [r["tilt"] for r in R], P.phase_color(ph), label=P.phase_label(ph))
    axes[0, 0].axhspan(-C.TILT_TOLERANCE_DEG, C.TILT_TOLERANCE_DEG, color="green", alpha=0.06)
    axes[0, 0].axhline(0, color="grey", lw=0.7)
    axes[0, 0].set(xlabel="Movement completed (%)", ylabel="Tilt (deg, + = P1 end higher)",
                   title="(a) Tilt time-course (mean +/- SD)")
    axes[0, 0].legend()
    P.series_by_trial(axes[0, 1], df, "max_abs_tilt", "Max |tilt| (deg)")
    axes[0, 1].set_title("(b) Tilt stability across trials")
    P.box_by_phase(axes[1, 0], {ph: df.loc[df.phase == ph, "pct_time_level"] for ph in phases}, phases,
                   f"% time |tilt| <= {C.TILT_TOLERANCE_DEG:g} deg", "(c) Time beam kept level")
    P.box_by_phase(axes[1, 1], {ph: df.loc[df.phase == ph, "n_tilt_corrections"] for ph in phases}, phases,
                   "Tilt reversals per trial", "(d) Number of balance corrections")
    fig.suptitle(f"{res['cfg']['id']} - Fig 5. Beam tilt", fontsize=11)
    P.save(fig, f"{fd}/fig05_beam_tilt.png")


def fig06_smoothness(res, fd):
    df, phases = res["df"], res["phases"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for a, col, t in [(axes[0], "sparc_P1", "(a) SPARC P1"), (axes[1], "sparc_P2", "(b) SPARC P2"),
                      (axes[2], "sparc_object", "(c) SPARC beam centre")]:
        P.box_by_phase(a, {ph: df.loc[df.phase == ph, col] for ph in phases}, phases, "SPARC", t)
    sub = {}
    for ph in phases:
        g = df[df.phase == ph]
        sub[ph] = (g["n_submov_P1"].astype(float) + g["n_submov_P2"].astype(float)) / 2
    P.box_by_phase(axes[3], sub, phases, "Speed peaks per player", "(d) Submovements (corrections)")
    fig.suptitle(f"{res['cfg']['id']} - Fig 6. Smoothness (SPARC: less negative = smoother)", fontsize=11)
    P.save(fig, f"{fd}/fig06_smoothness.png")


def fig07_forces(res, fd):
    df, phases = res["df"], res["phases"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ph in phases:
        R = _recs_of(res, ph)
        if not R:
            continue
        c = P.phase_color(ph)
        P.band(axes[0, 0], P.PCT, [np.abs(r["F1"]) for r in R], c, C.P1_LS, P.phase_label(ph))
        P.band(axes[0, 0], P.PCT, [np.abs(r["F2"]) for r in R], c, C.P2_LS)
        P.band(axes[0, 1], P.PCT, [r["Fk1"] for r in R], c, C.P1_LS, P.phase_label(ph))
        P.band(axes[0, 1], P.PCT, [r["Fk2"] for r in R], c, C.P2_LS)
    axes[0, 0].set(xlabel="Movement completed (%)", ylabel="|F| logged (N)",
                   title="(a) Logged commanded force |F1| solid, |F2| dashed")
    axes[0, 0].legend()
    axes[0, 1].axhline(0, color="grey", lw=0.7)
    axes[0, 1].set(xlabel="Movement completed (%)",
                   ylabel=f"m*a_x (N, m_eff={C.M_EFF_KG:g} kg)",
                   title="(b) Kinematic (inertial) force along path")
    P.box_by_phase(axes[0, 2], {ph: df.loc[df.phase == ph, "force_asym_idx"] for ph in phases}, phases,
                   "(|F1|-|F2|)/(|F1|+|F2|)", "(c) Logged load-sharing asymmetry", zero=True,
                   lead_note="> 0 : P1 carries more")
    P.box_by_phase(axes[1, 0], {ph: df.loc[df.phase == ph, "corr_force_share_idx"] for ph in phases}, phases,
                   "(I1-I2)/(I1+I2)", "(d) Corrective-force share (tilt-reducing m*a_y)", zero=True,
                   lead_note="> 0 : P1 corrects more")
    for i, ls in ((1, C.P1_LS), (2, C.P2_LS)):
        P.box_by_phase(axes[1, 1] if i == 1 else axes[1, 2],
                       {ph: df.loc[df.phase == ph, f"corr_force_impulse_P{i}"] for ph in phases}, phases,
                       "Impulse (N*s per kg)", f"({'e' if i == 1 else 'f'}) Corrective impulse P{i}")
    fig.suptitle(f"{res['cfg']['id']} - Fig 7. Forces (logged and kinematics-derived)", fontsize=11)
    P.save(fig, f"{fd}/fig07_forces.png")


LF_PANELS = [
    ("xcorr_lag_vel", "Lag (s)", "Velocity cross-correlation lag", True),
    ("frac_time_P1_leads", "Fraction", "Time-resolved: fraction of time P1 leads", 0.5),
    ("energy_leader_idx_kin", "Index", "Energy-flow leadership (coupling drags follower)", True),
    ("onset_lag_kin", "s", "Kinematic onset lag (P2 - P1)", True),
    ("peak_speed_lag", "s", "Peak-speed time lag (P2 - P1)", True),
    ("xcorr_lag_vy", "Lag (s)", "Vertical (balance) lead-lag", True),
    ("corr_force_share_idx", "Index", "Who corrects the tilt (force)", True),
    ("n_lead_switches", "Count", "Leadership switches per trial", False),
]


def fig08_leader_follower(res, fd):
    df, phases = res["df"], res["phases"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for a, (col, yl, t, z) in zip(axes.ravel(), LF_PANELS):
        if col in df:
            note = None if col == "n_lead_switches" else ("> 0.5 : P1 leads" if z == 0.5 else "> 0 : P1 leads/does more")
            P.box_by_phase(a, {ph: df.loc[df.phase == ph, col] for ph in phases}, phases, yl, t,
                           zero=z, lead_note=note)
    fig.suptitle(f"{res['cfg']['id']} - Fig 8. Leader-follower metrics (all signs: positive = Player 1 leads)",
                 fontsize=11)
    P.save(fig, f"{fd}/fig08_leader_follower.png")


def fig09_variability(res, fd):
    phases = res["phases"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ph in phases:
        R = _recs_of(res, ph)
        if len(R) < 2:
            continue
        c = P.phase_color(ph)
        for a, k, ls in [(axes[0], "x1", C.P1_LS), (axes[0], "x2", C.P2_LS),
                         (axes[1], "v1", C.P1_LS), (axes[1], "v2", C.P2_LS), (axes[2], "tilt", "-")]:
            a.plot(P.PCT, np.nanstd([r[k] for r in R], axis=0, ddof=1), color=c, ls=ls,
                   label=P.phase_label(ph) if ls == "-" else None)
    axes[0].set(title="(a) Position SD across trials", ylabel="SD x (m)", xlabel="Movement completed (%)")
    axes[1].set(title="(b) Velocity SD across trials", ylabel="SD v (m/s)", xlabel="Movement completed (%)")
    axes[2].set(title="(c) Tilt SD across trials", ylabel="SD tilt (deg)", xlabel="Movement completed (%)")
    axes[0].legend()
    el = res["el_df"]
    width = 0.38
    for j, ph in enumerate(phases):
        for k, part in enumerate(("early", "late")):
            v = el.loc[(el.phase == ph) & (el.part == part), "var_tilt"]
            if len(v):
                axes[3].bar(j + (k - 0.5) * width, v.values[0], width * 0.9, color=P.phase_color(ph),
                            alpha=0.4 if part == "early" else 0.9, edgecolor="k", lw=0.5,
                            label=part if j == 0 else None)
    axes[3].set_xticks(range(len(phases)))
    axes[3].set_xticklabels([P.phase_label(p) for p in phases], rotation=20, ha="right")
    axes[3].set(title=f"(d) Tilt variability: first {C.EARLY_N} vs last {C.LATE_N}", ylabel="Mean SD tilt (deg)")
    axes[3].legend()
    fig.suptitle(f"{res['cfg']['id']} - Fig 9. Trial-to-trial variability (P1 solid, P2 dashed)", fontsize=11)
    P.save(fig, f"{fd}/fig09_variability.png")


def fig10_early_late(res, fd):
    df, phases = res["df"], res["phases"]
    mets = [m for m in P.KEY_METRICS if m[0] in df.columns]
    nc = 4
    nr = int(np.ceil(len(mets) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.1 * nr), squeeze=False)
    width = 0.38
    for a, (col, lab, zero, note) in zip(axes.ravel(), mets):
        for j, ph in enumerate(phases):
            for k, part in enumerate(("early", "late")):
                v = df.loc[(df.phase == ph) & (df.early_late == part), col].astype(float).dropna()
                if len(v) == 0:
                    continue
                x = j + (k - 0.5) * width
                a.bar(x, v.mean(), width * 0.9, color=P.phase_color(ph), alpha=0.4 if part == "early" else 0.9,
                      edgecolor="k", lw=0.5, label=part if j == 0 else None)
                if len(v) > 1:
                    a.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color="k", capsize=2, lw=0.8)
        P._ref_line(a, zero)
        a.set_xticks(range(len(phases)))
        a.set_xticklabels(phases, rotation=20, fontsize=7)
        a.set_title(lab, fontsize=8.5)
    axes[0, 0].legend()
    for a in axes.ravel()[len(mets):]:
        a.axis("off")
    fig.suptitle(f"{res['cfg']['id']} - Fig 10. First {C.EARLY_N} vs last {C.LATE_N} trials of each phase "
                 "(mean +/- SEM over trials)", fontsize=11)
    P.save(fig, f"{fd}/fig10_early_vs_late.png")


FIGURES = [fig00_qc, fig01_trajectories, fig02_velocity, fig03_position_difference,
           fig04_across_trials, fig05_beam_tilt, fig06_smoothness, fig07_forces,
           fig08_leader_follower, fig09_variability, fig10_early_late]


# ======================================================================
# MAIN
# ======================================================================
def run_one(pcfg):
    out = os.path.join(C.RESULTS_ROOT, pcfg["id"])
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    print(f"\n{'=' * 64}\n{pcfg['id']}  ({pcfg['folder']})\n{'=' * 64}")
    res = process_pilot(pcfg)

    res["df"].to_csv(f"{out}/trial_summary.csv", index=False)
    res["phase_df"].to_csv(f"{out}/phase_summary.csv", index=False)
    res["el_df"].to_csv(f"{out}/early_late_summary.csv", index=False)
    res["qc"].to_csv(f"{out}/qc_per_trial.csv", index=False)
    with open(f"{out}/qc_report.txt", "w") as f:
        f.write(f"{pcfg['id']}  folder={pcfg['folder']}  protocol={pcfg['protocol']}\n\n")
        f.write("\n".join(res["log"]) + "\n")

    for fn in FIGURES:
        try:
            fn(res, fd)
        except Exception as e:   # one broken figure must not kill the rest
            print(f"  ! {fn.__name__} failed: {e!r}")
            plt.close("all")

    show = ["n_trials", "movement_time", "max_abs_tilt", "pct_time_level", "mean_abs_dx",
            "sparc_P1", "sparc_P2", "xcorr_lag_vel", "energy_leader_idx_kin", "corr_force_share_idx"]
    show = [c for c in show if c in res["phase_df"]]
    print("\n" + res["phase_df"].set_index("phase")[show].round(3).to_string())
    print(f"\n  -> {out}/")
    return res


def main(ids=None):
    for pcfg in C.PILOTS:
        if ids and pcfg["id"] not in ids:
            continue
        if not os.path.isdir(pcfg["folder"]):
            print(f"\nSKIP {pcfg['id']}: folder not found: {pcfg['folder']}")
            continue
        try:
            run_one(pcfg)
        except Exception as e:
            import traceback
            print(f"\nERROR in {pcfg['id']}: {e!r}")
            traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
