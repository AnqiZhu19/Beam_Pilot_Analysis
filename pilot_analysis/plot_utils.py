"""Shared plotting helpers (P1 = solid, P2 = dashed everywhere)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5, "axes.formatter.useoffset": False,
})

PCT = np.linspace(0, 100, C.N_GRID)


def phase_label(ph):
    return C.PHASE_LABELS.get(ph, ph)


def phase_color(ph):
    return C.PHASE_COLORS.get(ph, "grey")


def mean_sd(arr):
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr[None, :]
    m = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
    return m, sd


def mean_sem(arr):
    arr = np.asarray(arr, float)
    m, sd = mean_sd(arr)
    n = np.sum(np.isfinite(arr), axis=0) if arr.ndim == 2 else 1
    return m, sd / np.sqrt(np.maximum(n, 1))


def band(ax, x, arr, color, ls="-", label=None, alpha=0.18, lw=1.8, sem=False):
    """Mean +/- SD (or SEM) band. Returns n curves."""
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return 0
    m, e = (mean_sem if sem else mean_sd)(arr)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(x, m - e, m + e, color=color, alpha=alpha, lw=0)
    return arr.shape[0] if arr.ndim == 2 else 1


def _ref_line(ax, zero):
    """zero=True -> line at 0; a number -> line at that value."""
    if zero is True:
        ax.axhline(0, color="grey", lw=0.8, ls="--")
    elif zero is not False and zero is not None:
        ax.axhline(float(zero), color="grey", lw=0.8, ls="--")


def box_by_phase(ax, values_by_phase, phases, ylabel="", title="", zero=False,
                 point_colors=None, lead_note=None):
    """values_by_phase: {phase: array}. Boxes coloured by phase, raw points
    jittered on top (optionally coloured per point)."""
    rng = np.random.default_rng(0)
    for i, ph in enumerate(phases, start=1):
        v = np.asarray(values_by_phase.get(ph, []), float)
        ok = np.isfinite(v)
        if ok.sum() == 0:
            continue
        bp = ax.boxplot([v[ok]], positions=[i], widths=0.55, showmeans=True,
                        patch_artist=True, showfliers=False,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=4))
        bp["boxes"][0].set_facecolor(phase_color(ph))
        bp["boxes"][0].set_alpha(0.45)
        cols = None
        if point_colors is not None and ph in point_colors:
            cols = np.asarray(point_colors[ph])[ok]
        ax.scatter(i + rng.uniform(-0.15, 0.15, ok.sum()), v[ok], s=9,
                   color=cols if cols is not None else "k", alpha=0.6, zorder=3, lw=0)
    ax.set_xticks(range(1, len(phases) + 1))
    ax.set_xticklabels([phase_label(p) for p in phases], rotation=20, ha="right")
    ax.set_xlim(0.4, len(phases) + 0.6)
    _ref_line(ax, zero)
    if lead_note:
        ax.text(0.01, 0.99, lead_note, transform=ax.transAxes, va="top",
                fontsize=7, color="grey")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def shade_phases(ax, df, xcol="global_trial_idx"):
    """Shade the background by phase along a trial-index x axis."""
    for ph, g in df.groupby("phase", sort=False):
        ax.axvspan(g[xcol].min() - 0.5, g[xcol].max() + 0.5,
                   color=phase_color(ph), alpha=0.10, lw=0)


def series_by_trial(ax, df, col, ylabel, zero=False, fit=True, label_phases=False):
    shade_phases(ax, df)
    for ph, g in df.groupby("phase", sort=False):
        ax.plot(g["global_trial_idx"], g[col], "o-", ms=3, lw=0.8,
                color=phase_color(ph), label=phase_label(ph) if label_phases else None)
        if fit:
            ok = np.isfinite(g[col].values.astype(float))
            if ok.sum() >= 3:
                x = g["global_trial_idx"].values[ok]
                pfit = np.polyfit(x, g[col].values[ok].astype(float), 1)
                ax.plot(x, np.polyval(pfit, x), color=phase_color(ph), lw=2, alpha=0.6)
    _ref_line(ax, zero)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Trial (chronological)")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Metrics shown in summary grids: (column, axis label, zero line?, lead note)
LEAD = "> 0 : P1 leads"
KEY_METRICS = [
    ("movement_time", "Movement time (s)", False, None),
    ("max_abs_tilt", "Max |beam tilt| (deg)", False, None),
    ("rms_tilt", "RMS beam tilt (deg)", False, None),
    ("pct_time_level", f"% time |tilt| <= {C.TILT_TOLERANCE_DEG:g} deg", False, None),
    ("mean_abs_dx", "Mean |x1 - x2| (m)", False, None),
    ("sparc_P1", "SPARC P1 (closer to 0 = smoother)", False, None),
    ("sparc_P2", "SPARC P2", False, None),
    ("sparc_object", "SPARC beam centre", False, None),
    ("xcorr_lag_vel", "Velocity x-corr lag (s)", True, LEAD),
    ("frac_time_P1_leads", "Fraction of time P1 leads", 0.5, "> 0.5 : P1 leads"),
    ("energy_leader_idx_kin", "Energy leadership index", True, LEAD),
    ("Fnc_share_y", "Corrective-force share (Eq. 7, y)", True, "> 0 : P1 corrects more"),
    ("Fint_lead_x", "Interaction force along motion (N)", True, "> 0 : P1 pulls P2 (leads)"),
    ("dominance_P1_x", "Force dominance P1 (path axis)", 0.5, "> 0.5 : P1 dominant"),
    ("crp_mean_deg", "Mean relative phase (deg)", True, LEAD),
    ("onset_lag_kin", "Onset lag P2 - P1 (s)", True, LEAD),
]


# ======================================================================
# SHARED FIGURE SPECIFICATIONS
# Each spec is drawn per dyad (run_analysis.py: phases overlaid, boxes) and
# across dyads (overlay_analysis.py: dyads overlaid per phase) - so both
# scripts always show the same metrics.
#   curves : (axis label, [(record key, linestyle, legend label), ...])
#   metrics: (trial_summary column, label, reference line, note)
# ======================================================================
P1, P2 = C.P1_LS, C.P2_LS
FIG_SPECS = [
    dict(name="fig02_velocity", title="Velocity",
         curves=[("Hand velocity x (m/s)", [("v1", P1, "P1"), ("v2", P2, "P2")]),
                 ("Relative velocity v1 - v2 (m/s)", [("dv", "-", "v1 - v2")]),
                 ("Beam-centre velocity (m/s)", [("obj_v", "-", "(v1 + v2)/2")])],
         metrics=[("peak_speed_P1", "Peak speed P1 (m/s)", False, None),
                  ("peak_speed_P2", "Peak speed P2 (m/s)", False, None),
                  ("mean_speed_P1", "Mean speed P1 (m/s)", False, None),
                  ("mean_speed_P2", "Mean speed P2 (m/s)", False, None),
                  ("peak_speed_lag", "Peak-speed time lag P2 - P1 (s)", True, LEAD)]),
    dict(name="fig03_position_difference", title="Hand-position difference",
         curves=[("x1 - x2 (m)", [("dx", "-", "x1 - x2")])],
         metrics=[("mean_abs_dx", "Mean |x1 - x2| (m)", False, None),
                  ("peak_abs_dx", "Peak |x1 - x2| (m)", False, None),
                  ("rms_dx", "RMS x1 - x2 (m)", False, None),
                  ("mean_dx", "Mean signed x1 - x2 (m)", True, "> 0 : P1 further along +x"),
                  ("frac_opposing_motion", "Fraction of time moving apart", False, None),
                  ("traj_dev_P1", "Path deviation from Baseline 1, P1 (m)", False, None),
                  ("traj_dev_P2", "Path deviation from Baseline 1, P2 (m)", False, None)]),
    dict(name="fig05_beam_tilt", title="Beam tilt",
         curves=[("Beam tilt (deg, + = P1 end higher)", [("tilt", "-", "tilt")]),
                 ("Vertical hand velocity (m/s)", [("vy1", P1, "P1"), ("vy2", P2, "P2")])],
         metrics=[("max_abs_tilt", "Max |tilt| (deg)", False, None),
                  ("rms_tilt", "RMS tilt (deg)", False, None),
                  ("mean_tilt", "Mean signed tilt (deg)", True, "> 0 : P1 end higher"),
                  ("sd_tilt", "Within-trial SD of tilt (deg)", False, None),
                  ("pct_time_level", f"% time |tilt| <= {C.TILT_TOLERANCE_DEG:g} deg", False, None),
                  ("n_tilt_corrections", "Tilt reversals (corrections)", False, None),
                  ("sparc_tilt_rate", "SPARC of tilt rate", False, None)]),
    dict(name="fig06_smoothness", title="Smoothness (SPARC: closer to 0 = smoother)",
         curves=[],
         metrics=[("sparc_P1", "SPARC P1", False, None), ("sparc_P2", "SPARC P2", False, None),
                  ("sparc_object", "SPARC beam centre", False, None),
                  ("n_submov_P1", "Speed peaks P1", False, None),
                  ("n_submov_P2", "Speed peaks P2", False, None),
                  ("path_len_P1", "Path length P1 (m)", False, None),
                  ("path_len_P2", "Path length P2 (m)", False, None)]),
    dict(name="fig07_forces",
         title="Forces, viscous-field model (signed; + = towards +axis)",
         curves=[("Interaction force P1 -> P2, path axis (N)", [("Fintx", "-", "F_int x")]),
                 ("Interaction force P1 -> P2, vertical (N)", [("Finty", "-", "F_int y")]),
                 ("Participant force, path axis (N)", [("Fh1x", P1, "P1"), ("Fh2x", P2, "P2")]),
                 ("Participant force, vertical (N)", [("Fh1y", P1, "P1"), ("Fh2y", P2, "P2")]),
                 ("Effort vs viscous field B*v (N)", [("Fv1x", P1, "P1"), ("Fv2x", P2, "P2")]),
                 ("Logged commanded F1 / F2 (N)", [("F1", P1, "F1"), ("F2", P2, "F2")])],
         metrics=[("mean_Fint_x", "Mean interaction force x (N)", True, "> 0 : P1 pushes P2 towards +x"),
                  ("mean_abs_Fint_x", "Mean |interaction force| x (N)", False, None),
                  ("Fint_lead_x", "Interaction force along motion (N)", True, "> 0 : P1 pulls P2 (leads)"),
                  ("frac_P1_pulls_x", "Fraction of interaction time P1 pulls", 0.5, "> 0.5 : P1 leads"),
                  ("pct_interaction_x", "% time interacting (x)", False, None),
                  ("mean_Fint_y", "Mean interaction force y (N)", True, "> 0 : P1 pushes P2 up"),
                  ("mean_abs_Fint_y", "Mean |interaction force| y (N)", False, None),
                  ("mean_Fh_x_P1", "Mean participant force x, P1 (N)", True, None),
                  ("mean_Fh_x_P2", "Mean participant force x, P2 (N)", True, None),
                  ("mean_Fh_y_P1", "Mean participant force y, P1 (N)", True, None),
                  ("mean_Fh_y_P2", "Mean participant force y, P2 (N)", True, None),
                  ("work_x_P1", "Work on system x, P1 (J)", True, "> 0 drives, < 0 brakes"),
                  ("work_x_P2", "Work on system x, P2 (J)", True, "> 0 drives, < 0 brakes"),
                  ("dominance_P1_x", "Force dominance P1, x", 0.5, "> 0.5 : P1 dominant"),
                  ("mean_F1", "Mean F1 logged (N)", True, None),
                  ("mean_F2", "Mean F2 logged (N)", True, None)]),
    dict(name="fig07b_corrective_forces",
         title="Corrective forces, Eq. 7: mean |F(t) - SMA(F)(t)|",
         curves=[("F - SMA(F), vertical (N)", [("Fnc1y", P1, "P1"), ("Fnc2y", P2, "P2")]),
                 ("F - SMA(F), path axis (N)", [("Fnc1x", P1, "P1"), ("Fnc2x", P2, "P2")])],
         metrics=[("Fnc_y_P1", "Corrective force y, P1 (N)", False, None),
                  ("Fnc_y_P2", "Corrective force y, P2 (N)", False, None),
                  ("Fnc_share_y", "Corrective share y", True, "> 0 : P1 corrects more"),
                  ("Fnc_int_y", "Corrective interaction force y (N)", False, "dyad value"),
                  ("Fnc_x_P1", "Corrective force x, P1 (N)", False, None),
                  ("Fnc_x_P2", "Corrective force x, P2 (N)", False, None),
                  ("Fnc_share_x", "Corrective share x", True, "> 0 : P1 corrects more"),
                  ("Fnc_int_x", "Corrective interaction force x (N)", False, "dyad value")]),
    dict(name="fig08_leader_follower", title="Leader-follower (all: positive = Player 1 leads)",
         curves=[],
         metrics=[("xcorr_lag_vel", "Velocity x-corr lag (s)", True, LEAD),
                  ("frac_time_P1_leads", "Fraction of time P1 leads", 0.5, "> 0.5 : P1 leads"),
                  ("n_lead_switches", "Leadership switches / trial", False, None),
                  ("energy_leader_idx_kin", "Energy-flow leadership", True, LEAD),
                  ("onset_lag_kin", "Kinematic onset lag P2 - P1 (s)", True, LEAD),
                  ("init_time_diff", "Leave-home lag P2 - P1 (s)", True, LEAD),
                  ("target_time_diff", "Target-arrival lag P2 - P1 (s)", True, "> 0 : P1 arrives first"),
                  ("peak_speed_lag", "Peak-speed lag P2 - P1 (s)", True, LEAD),
                  ("xcorr_lag_vy", "Vertical velocity x-corr lag (s)", True, LEAD),
                  ("crp_mean_deg", "Mean relative phase (deg)", True, LEAD),
                  ("Fint_lead_x", "Interaction force along motion (N)", True, "> 0 : P1 pulls P2"),
                  ("Fnc_share_y", "Corrective share y (Eq. 7)", True, "> 0 : P1 corrects more")]),
    dict(name="fig13_cross_correlation",
         title="Lead-lag cross-correlation r(tau) (peak at tau > 0 : P1 leads by tau)",
         curves=[("r, horizontal velocity", [("xc_vel", "-", "v1 vs v2")], "lag"),
                 ("r, vertical velocity", [("xc_vy", "-", "vy1 vs vy2")], "lag")],
         metrics=[("xcorr_lag_vel", "Peak lag, horizontal (s)", True, LEAD),
                  ("xcorr_lag_vel_pct", "Peak lag, % of movement time", True, LEAD),
                  ("xcorr_r_vel", "Peak correlation, horizontal", False, "1 = identical shape"),
                  ("xcorr_lag_vy", "Peak lag, vertical (s)", True, LEAD),
                  ("xcorr_r_vy", "Peak correlation, vertical", False, None),
                  ("frac_time_P1_leads", "Fraction of time P1 leads (windowed)", 0.5, "> 0.5 : P1 leads"),
                  ("n_lead_switches", "Leadership switches / trial", False, None)]),
    dict(name="fig11_relative_phase", title="Relative phase (temporal coordination)",
         curves=[("Continuous relative phase (deg)", [("crp", "-", "phase P1 - phase P2")])],
         metrics=[("crp_mean_deg", "Mean CRP, path axis (deg)", True, LEAD),
                  ("crp_sd_deg", "CRP variability (circular SD, deg)", False, "lower = more stable"),
                  ("relphase_vel_deg", "Hilbert rel. phase, velocity (deg)", True, LEAD),
                  ("plv_vel", "Phase locking, velocity (0-1)", False, None),
                  ("relphase_vy_deg", "Hilbert rel. phase, vertical (deg)", True, LEAD),
                  ("plv_vy", "Phase locking, vertical (0-1)", False, None)]),
]


def _grid(sf, n, ncol):
    nr = max(1, int(np.ceil(n / ncol)))
    axes = sf.subplots(nr, ncol, squeeze=False)
    for a in axes.ravel()[n:]:
        a.axis("off")
    return axes.ravel()[:n]


def _note(ax, note):
    if note:
        ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", fontsize=6.5, color="grey")


def _figure(n_curve_rows, curve_h, n_metrics, ncol, width):
    m_rows = int(np.ceil(n_metrics / ncol)) if n_metrics else 0
    h_top, h_bot = n_curve_rows * curve_h, m_rows * 2.9
    fig = plt.figure(figsize=(width, h_top + h_bot + 0.6), layout="constrained")
    if n_curve_rows and m_rows:
        top, bot = fig.subfigures(2, 1, height_ratios=[h_top, h_bot])
    elif n_curve_rows:
        top, bot = fig, None
    else:
        top, bot = None, fig
    return fig, top, bot


def _curve_x(entry):
    """x values and label for a curve entry (ylabel, keys[, 'lag'])."""
    if len(entry) > 2 and entry[2] == "lag":
        from metrics import XC_LAGS
        return XC_LAGS, "Lag tau (s)  (> 0 : P1 leads)"
    return PCT, "Movement completed (%)"


def _has_data(group, recs_of, keys):
    """True if any trial of any dyad has finite values for any of the keys."""
    for res in group:
        for ph in res["phases"]:
            for r in recs_of(res, ph):
                for key, *_ in keys:
                    v = r.get(key)
                    if v is not None and np.ndim(v) == 1 and np.isfinite(v).any():
                        return True
    return False


def render_pilot_spec(spec, res, recs_of, path):
    """Per-dyad: one panel per curve (phases in their colours, mean +/- SD
    across trials), then one box plot per metric (dots = trials)."""
    df, phases = res["df"], res["phases"]
    ncol = 3
    curves = [c for c in spec["curves"] if _has_data([res], recs_of, c[1])]
    metrics = [m for m in spec["metrics"] if m[0] in df.columns and df[m[0]].notna().any()]
    fig, top, bot = _figure(int(np.ceil(len(curves) / ncol)), 3.0, len(metrics), 4, 17)
    if curves:
        for ax, entry in zip(_grid(top, len(curves), ncol), curves):
            ylab, keys = entry[0], entry[1]
            X, xlab = _curve_x(entry)
            for ph in phases:
                R = recs_of(res, ph)
                for key, ls, lab in keys:
                    arr = [r[key] for r in R if key in r]
                    if arr and np.isfinite(arr).any():
                        band(ax, X, arr, phase_color(ph), ls,
                             label=phase_label(ph) if key == keys[0][0] else None)
            ax.axhline(0, color="grey", lw=0.6)
            if X is not PCT:
                ax.axvline(0, color="grey", lw=0.6)
            ax.set(xlabel=xlab, ylabel=ylab)
            if len(keys) > 1:
                _note(ax, "   ".join(f"{'—' if ls == '-' else ('- -' if ls == '--' else '···')} {lab}"
                                     for _, ls, lab in keys))
        if top.axes[0].get_legend_handles_labels()[0]:
            top.axes[0].legend(fontsize=6.5, loc="lower right")
    if metrics:
        for ax, (col, lab, ref, note) in zip(_grid(bot, len(metrics), 4), metrics):
            box_by_phase(ax, {ph: df.loc[df.phase == ph, col] for ph in phases}, phases,
                         "", lab, zero=ref)
            ax.title.set_fontsize(8.5)
            ax.tick_params(axis="x", labelsize=7)
            _note(ax, note)
    fig.suptitle(f"{res['cfg']['id']} - {spec['title']}   (curves: mean +/- SD over trials; "
                 "boxes: trials)", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


PROTO_MARKERS = {"A": "s", "B": "o"}


def dodge_metric(ax, group, col, phases, ref=False, note=None, title="", show_mean=True):
    """Overlay of a per-trial metric across dyads: for every phase each dyad
    is a coloured point (mean over its trials) +/- SD over its trials,
    side by side; black diamond = mean of dyad means +/- SEM across dyads."""
    n = len(group)
    width = 0.7
    offs = (np.arange(n) - (n - 1) / 2) * (width / max(n, 1)) if n > 1 else np.zeros(1)
    for j, ph in enumerate(phases):
        means = []
        for k, res in enumerate(group):
            d = res["df"]
            if col not in d:
                continue
            v = d.loc[d.phase == ph, col].astype(float).dropna()
            if len(v) == 0:
                continue
            means.append(v.mean())
            ax.errorbar(j + offs[k] - 0.08, v.mean(), v.std(ddof=1) if len(v) > 1 else 0,
                        color=res["color"], marker=PROTO_MARKERS.get(res["cfg"]["protocol"], "o"),
                        ms=4, lw=1, capsize=1.5,
                        label=res["cfg"]["id"] if j == 0 else None)
        if show_mean and len(means) > 1:
            mu = np.mean(means)
            se = np.std(means, ddof=1) / np.sqrt(len(means))
            ax.errorbar(j + width / 2 + 0.02, mu, se, color="k", marker="D", ms=5, lw=2,
                        capsize=3, zorder=5, label="mean of dyads +/- SEM" if j == 0 else None)
        ax.axvspan(j - 0.5, j + 0.5, color=phase_color(ph), alpha=0.07, lw=0)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, rotation=20, fontsize=7)
    ax.set_xlim(-0.5, len(phases) - 0.5)
    _ref_line(ax, ref)
    _note(ax, note)
    ax.set_title(title, fontsize=8.5)


def overlay_dyad_curve(res, recs_of, ph, key):
    R = recs_of(res, ph)
    arr = [r[key] for r in R if key in r and np.ndim(r[key]) == 1]
    if not arr or not np.isfinite(arr).any():
        return None
    return np.nanmean(arr, axis=0)


def render_overlay_spec(spec, group, phases, recs_of, tag, path, show_mean=True):
    """Across dyads: curve rows x phase columns (thin = dyad mean, black =
    mean of dyads +/- SEM), then the same metrics as the per-dyad figure."""
    curves = [c for c in spec["curves"] if _has_data(group, recs_of, c[1])]
    metrics = [m for m in spec["metrics"]
               if any(m[0] in r["df"] and r["df"][m[0]].notna().any() for r in group)]
    ncol_c = len(phases)
    fig, top, bot = _figure(len(curves), 2.7, len(metrics), 4, max(17, 4.2 * ncol_c))
    if curves:
        axes = top.subplots(len(curves), ncol_c, squeeze=False, sharey="row")
        for i, entry in enumerate(curves):
            ylab, keys = entry[0], entry[1]
            X, xlab = _curve_x(entry)
            for j, ph in enumerate(phases):
                ax = axes[i, j]
                for key, ls, lab in keys:
                    allc = []
                    for res in group:
                        c = overlay_dyad_curve(res, recs_of, ph, key)
                        if c is None:
                            continue
                        allc.append(c)
                        ax.plot(X, c, color=res["color"], ls=ls, lw=1.1, alpha=0.85,
                                label=res["cfg"]["id"] if (i == 0 and j == 0 and key == keys[0][0]) else None)
                    if show_mean and len(allc) > 1:
                        band(ax, X, allc, "k", ls, lw=2.2, alpha=0.14, sem=True,
                             label="mean of dyads +/- SEM" if (i == 0 and j == 0 and key == keys[0][0]) else None)
                ax.axhline(0, color="grey", lw=0.6)
                if i == 0:
                    ax.set_title(phase_label(ph))
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                    if len(keys) > 1:
                        _note(ax, "   ".join(f"{'—' if ls == '-' else ('- -' if ls == '--' else '···')} {lab}"
                                             for _, ls, lab in keys))
                if X is not PCT:
                    ax.axvline(0, color="grey", lw=0.6)
                ax.set_xlabel(xlab, fontsize=8)
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend(fontsize=6.5, loc="lower right")
    if metrics:
        maxes = _grid(bot, len(metrics), 4)
        for ax, (col, lab, ref, note) in zip(maxes, metrics):
            dodge_metric(ax, group, col, phases, ref, note, lab, show_mean)
        maxes[0].legend(fontsize=6, loc="best")
    fig.suptitle(f"{spec['title']} - {tag}\n(dots: dyad mean +/- SD over its trials; "
                 "black: mean of dyads +/- SEM)", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# REAL-TIME (completion time) TRAJECTORIES
# ======================================================================
RT_ROWS = [("Hand position x (m)", [("rt_x1", P1), ("rt_x2", P2)]),
           ("Hand velocity x (m/s)", [("rt_v1", P1), ("rt_v2", P2)]),
           ("Beam tilt (deg)", [("rt_tilt", "-")])]


def rt_mean(arr):
    """Mean over trials where at least REALTIME_MIN_FRAC of them still run."""
    arr = np.asarray(arr, float)
    n = np.sum(np.isfinite(arr), axis=0)
    m = np.nanmean(arr, axis=0)
    m[n < C.REALTIME_MIN_FRAC * arr.shape[0]] = np.nan
    return m


def rt_time_axis():
    return np.arange(0, C.REALTIME_MAX_S, 1.0 / C.REALTIME_FS)


# ======================================================================
# WINDOWED LEAD-LAG HEATMAP
# ======================================================================
def tlcc_panel(ax, mat, title):
    """mat: (n_lags, n_windows) mean windowed cross-correlation."""
    from metrics import TLCC_LAGS
    n_win = mat.shape[1]
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[0, 100, TLCC_LAGS[0], TLCC_LAGS[-1]])
    # peak lag per window
    with np.errstate(all="ignore"):
        ok = np.isfinite(mat).any(axis=0)
        pk = np.full(n_win, np.nan)
        pk[ok] = TLCC_LAGS[np.nanargmax(np.where(np.isfinite(mat[:, ok]), mat[:, ok], -9), axis=0)]
    centers = (np.arange(n_win) + 0.5) * 100 / n_win
    ax.plot(centers, pk, "k.-", lw=1, ms=4, label="peak lag")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set(xlabel="Movement completed (%)", ylabel="Lag tau (s)  (> 0 : P1 leads)", title=title)
    return im
