"""
Per-dyad analysis.  One output folder per pilot:

    Results/<id>/
        trial_summary.csv      one row per trial, all metrics
        phase_summary.csv      one row per phase (means, variability, learning slopes)
        early_late_summary.csv first vs last N trials of every phase
        qc_report.txt          clock sync, data gaps, sanity checks
        figures/fig00 ... fig11 (fig02/03/05-08/11 share their metric lists
                               with overlay_analysis.py via plot_utils.FIG_SPECS)

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

# trial-table bookkeeping columns that are not metrics
NON_METRIC = {"block", "trial", "global_trial_idx", "trial_in_block", "trial_in_phase",
              "n_in_phase", "phase_idx", "t_begin", "t_end", "t_move_start_event", "t_running",
              "t_prim_home", "t_sec_home", "t_prim_target", "t_sec_target",
              "onset_P1", "onset_P2", "offset_P1", "offset_P2",
              "t_peak_speed_P1", "t_peak_speed_P2"}


def metric_columns(df):
    """Every numeric per-trial metric (averaged per phase in phase_summary)."""
    return [c for c in df.columns if c not in NON_METRIC and pd.api.types.is_numeric_dtype(df[c])]


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

    # viscous-field force model: gains from config or estimated from the logs
    fit = U.fit_force_model(p, tt)
    for ax, f in fit.items():
        say(f"force-model fit, {ax}-axis: coupling K_c = {f['K_c']:.1f} N/m, B_c = {f['B_c']:.2f} N s/m "
            f"(R^2 {f['r2_c']:.2f}, target {f['src_c']}); viscous B = {f['B_visc']:.2f} N s/m "
            f"(R^2 {f['r2_v']:.2f}, from {C.COMMANDED_FORCE_COLS.get(ax)})")
    gains = U.resolve_gains(fit)
    for ax, g in gains.items():
        say(f"USED {ax}-axis: B_visc = {g['B_visc']:.2f} [{g['src_visc']}], "
            f"K_c = {g['K_c']:.1f}, B_c = {g['B_c']:.2f} [{g['src_c']}]")

    # Full-resolution segments are NOT kept (memory): every trial is reduced
    # to scalar metrics + 101-point normalised curves; one example segment is
    # kept for the QC figure.
    segs, recs, rows, qcs = {}, {}, [], []
    example_idx = len(tt) // 2
    n_long = 0
    for n, (_, row) in enumerate(tt.iterrows()):
        key = (row["block"], row["trial"])
        out = row.to_dict()
        if row["t_end"] - row["t_begin"] > C.MAX_TRIAL_S:
            n_long += 1
            qcs.append(dict(block=row["block"], trial=row["trial"], skipped="too long"))
            rows.append(out)
            continue
        d, qc = U.segment_trial(row, p, s)
        qc.update(block=row["block"], trial=row["trial"])
        qcs.append(qc)
        if d is not None:
            mtr, (t0, t1) = M.trial_metrics(d, gains)
            out.update(mtr)
            recs[key] = M.normalized_record(d, t0, t1)
            if n >= example_idx and not segs:
                segs[key] = d
        rows.append(out)
        del d
    del p, s, trial          # raw logs no longer needed
    df = pd.DataFrame(rows)
    qc = pd.DataFrame(qcs)
    say(f"{len(recs)}/{len(tt)} trials segmented")
    if n_long:
        say(f"  WARNING: {n_long} trials longer than {C.MAX_TRIAL_S:g} s were skipped "
            "(missing 'end' event?) - see qc_per_trial.csv")

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
    for ax in ("x", "y"):
        col = f"mass_limit_kg_{ax}"
        if col in df and df[col].notna().any():
            say(f"mass term negligible on {ax} only if the robot's apparent mass << "
                f"{df[col].median():.2f} kg (= B v / a at the 90th percentile)")
    if "Fnc_y_from_logged_F" in df and df["Fnc_y_from_logged_F"].notna().any():
        say("NOTE: no vertical force-model gains -> corrective force y (Eq. 7) computed from the "
            "logged F1/F2 instead. Set VISCOUS_B['y'] / COUPLING['y'] to use the model.")
    if not any(np.isfinite(g["K_c"]) for g in gains.values()):
        say("WARNING: no coupling gains -> interaction force not computed. "
            "Set config.COUPLING to the values in your controller.")
    zero_tilt = df["mean_tilt"].median()
    say(f"median signed tilt across trials: {zero_tilt:.2f} deg"
        + ("  <-- large: check y-offset between robots / SECONDARY_Y_SOURCE" if abs(zero_tilt) > 3 else ""))

    phase_df, el_df = phase_summaries(df, recs, pcfg)
    return dict(cfg=pcfg, tt=tt, df=df, segs=segs, recs=recs, qc=qc, gains=gains,
                phase_df=phase_df, el_df=el_df, log=log,
                phases=[ph for ph in C.PROTOCOLS[pcfg["protocol"]]["phases"] if ph in set(df["phase"])])


def _phase_recs(df, recs, phase, subset=None):
    g = df[df["phase"] == phase]
    if subset is not None:
        g = g[g["early_late"] == subset]
    return [recs[(b, t)] for b, t in zip(g["block"], g["trial"]) if (b, t) in recs]


def phase_summaries(df, recs, pcfg):
    rows, el_rows = [], []
    cols = metric_columns(df)
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


def fig01b_trajectories_realtime(res, fd):
    """Trajectories in real time from movement onset (not normalised):
    thin = trials, thick = mean while >= 50 % of trials are still moving,
    vertical line + band = mean +/- SD movement (completion) time."""
    df, phases = res["df"], res["phases"]
    t = P.rt_time_axis()
    fig, axes = plt.subplots(len(P.RT_ROWS), len(phases), figsize=(4.2 * len(phases), 8.5),
                             sharey="row", sharex=True, squeeze=False)
    tmax = 0
    for j, ph in enumerate(phases):
        R = _recs_of(res, ph)
        mt = df.loc[df.phase == ph, "movement_time"].astype(float).dropna()
        if not R:
            continue
        for i, (ylab, keys) in enumerate(P.RT_ROWS):
            ax = axes[i, j]
            for key, ls in keys:
                arr = np.array([r[key] for r in R if key in r])
                if arr.size == 0:
                    continue
                for row_ in arr:
                    ax.plot(t, row_, color=P.phase_color(ph), lw=0.5, alpha=0.35, ls=ls)
                ax.plot(t, P.rt_mean(arr), color="k", lw=2, ls=ls)
            if len(mt):
                ax.axvspan(mt.mean() - mt.std(), mt.mean() + mt.std(), color="grey", alpha=0.15, lw=0)
                ax.axvline(mt.mean(), color="k", lw=1, ls=":")
                tmax = max(tmax, mt.max())
            ax.axhline(0, color="grey", lw=0.5)
            if j == 0:
                ax.set_ylabel(ylab)
        if len(mt):
            axes[0, j].set_title(f"{P.phase_label(ph)}\ncompletion time {mt.mean():.2f} +/- {mt.std():.2f} s "
                                 f"(n={len(mt)})", fontsize=9)
        axes[-1, j].set_xlabel("Time from movement onset (s)")
    if tmax:
        axes[0, 0].set_xlim(0, min(C.REALTIME_MAX_S, tmax * 1.05))
    fig.suptitle(f"{res['cfg']['id']} - Fig 1b. Trajectories in real time (P1 solid, P2 dashed; "
                 "dotted line + grey band = mean +/- SD completion time)", fontsize=11)
    P.save(fig, f"{fd}/fig01b_trajectories_realtime.png")


def fig13b_lead_lag_heatmap(res, fd):
    """Windowed cross-correlation of the horizontal velocities along the
    movement (mean over trials): where in the movement does who lead?"""
    phases = res["phases"]
    fig, axes = plt.subplots(1, len(phases), figsize=(4.4 * len(phases), 4.2), squeeze=False,
                             layout="constrained")
    im = None
    for j, ph in enumerate(phases):
        mats = [r["tlcc_vel"] for r in _recs_of(res, ph) if "tlcc_vel" in r]
        if mats:
            im = P.tlcc_panel(axes[0, j], np.nanmean(mats, axis=0), P.phase_label(ph))
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="r", shrink=0.8)
    fig.suptitle(f"{res['cfg']['id']} - Fig 13b. Lead-lag along the movement (windowed cross-correlation; "
                 "black = peak lag per window)", fontsize=11)
    fig.savefig(f"{fd}/fig13b_lead_lag_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def spec_figures(res, fd):
    for spec in P.FIG_SPECS:
        try:
            P.render_pilot_spec(spec, res, _recs_of, f"{fd}/{spec['name']}.png")
        except Exception as e:
            print(f"  ! {spec['name']} failed: {e!r}")
            plt.close("all")


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


FIGURES = [fig00_qc, fig01_trajectories, fig01b_trajectories_realtime, fig04_across_trials,
           fig09_variability, fig10_early_late, fig13b_lead_lag_heatmap, spec_figures]


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
            "sparc_P1", "sparc_P2", "xcorr_lag_vel", "Fint_lead_x", "Fnc_share_y"]
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
