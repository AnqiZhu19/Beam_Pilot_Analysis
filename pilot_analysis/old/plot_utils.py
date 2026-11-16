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
    ("corr_force_share_idx", "Corrective-force share", True, "> 0 : P1 corrects more"),
    ("force_asym_idx", "Logged force asymmetry", True, "> 0 : |F1| > |F2|"),
    ("onset_lag_kin", "Onset lag P2 - P1 (s)", True, LEAD),
]
