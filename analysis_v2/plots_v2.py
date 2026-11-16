"""Figures for the new pilot analysis. P1 = blue solid, P2 = orange dashed."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import config_v2 as C
import segmentation as S
import metrics_v2 as M

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7", "axes.formatter.useoffset": False,
})
B, BC = C.BLOCK_ORDER, C.BLOCK_COLORS
PCT = np.linspace(0, 100, C.N_GRID)

SHADE = {
    "vert": {1: "#c9e6d8", 0: "#f6f3dc", -1: "#f8d9a0"},
    "inter": {"stationary": "#f6f3dc", "single": "#c9e6d8", "aligned": "#c8c2f0", "opposed": "#f4b9b9"},
    "beam": {1: "#c8c2f0", 0: "#f6f3dc", -1: "#f6cddd"},
    "phase": {"acc": "#e3ecf9", "dec": "#fbe3d6", "final": "#dff1df"},
}
INTER_COLORS = {"stationary": "#c3c2b7", "single": "#1baf7a", "aligned": "#6250d6", "opposed": "#e34948"}


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _mean_sd(arr):
    arr = np.asarray(arr, float)
    return np.nanmean(arr, 0), np.nanstd(arr, 0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])


def band(ax, x, arr, color, ls="-", label=None, min_frac=0.0):
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return
    m, sd = _mean_sd(arr)
    if min_frac > 0:
        ok = np.mean(np.isfinite(arr), 0) >= min_frac
        m, sd = np.where(ok, m, np.nan), np.where(ok, sd, np.nan)
    ax.plot(x, m, color=color, ls=ls, lw=1.8, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, lw=0)


def shade_runs(ax, t, lab, cmap):
    for a, b, v in S.runs(lab):
        ax.axvspan(t[a], t[b - 1], color=cmap[v], lw=0, zorder=0)


def _curves(res, curves, block, key, part=None):
    g = res[res["block"] == block]
    if part:
        g = g[g["early_late"] == part]
    return [curves[(r.block_num, r.trial)][key] for r in g.itertuples()]


# ======================================================================
# F0  Fig. 8 replica
# ======================================================================
def fig_example_trial(d, lab, inter, m, title, path):
    t = d["t"].values
    fig, ax = plt.subplots(6, 1, figsize=(8.5, 11), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1, 1.2, 1.2, 1.2, 1]))
    for i, (k, col, who) in enumerate((("vy1", C.P1_COLOR, "P1"), ("vy2", C.P2_COLOR, "P2"))):
        shade_runs(ax[i], t, lab[f"v{i + 1}"], SHADE["vert"])
        ax[i].plot(t, d[k], color=col, lw=1.3)
        ax[i].axhline(0, color="k", lw=1.2)
        ax[i].set_ylabel(f"$\\dot y_{{{who}}}$ (m/s)")
    ax[0].set_title(f"(a) Vertical hand movement segmentation (HMM)   -   {title}", loc="left")
    ax[0].legend(handles=[Patch(edgecolor="#8a8a85", facecolor=SHADE["vert"][v], label=S.STATE_NAMES["vert"][v]) for v in (0, 1, -1)],
                 ncol=3, loc="upper right", frameon=False)

    shade_runs(ax[2], t, np.array([S.INTERACTION.index(s) for s in inter]),
               {i: SHADE["inter"][k] for i, k in enumerate(S.INTERACTION)})
    ax[2].plot(t, 1e3 * (d["y1"] - d["y1"].iloc[0]), color=C.P1_COLOR, lw=1.5, label="P1")
    ax[2].plot(t, 1e3 * (d["y2"] - d["y2"].iloc[0]), color=C.P2_COLOR, lw=1.5, ls="--", label="P2")
    ax[2].set_ylabel("hand height\n(mm, re start)")
    ax[2].set_title("(b) Interaction dynamics", loc="left")
    ax[2].legend(handles=[Patch(edgecolor="#8a8a85", facecolor=SHADE["inter"][k], label=k) for k in S.INTERACTION]
                 + [Line2D([], [], color=C.P1_COLOR, label="P1"), Line2D([], [], color=C.P2_COLOR, ls="--", label="P2")],
                 ncol=6, loc="upper right", frameon=False)

    shade_runs(ax[3], t, lab["beam"], SHADE["beam"])
    ax[3].plot(t, d["theta"], color=C.OBJ_COLOR, lw=1.5)
    ax[3].axhline(0, color="k", lw=1.2)
    ax[3].set_ylabel("beam tilt θ (deg)")
    ax[3].set_title("(c) Beam rotation segmentation (HMM on dθ/dt)", loc="left")
    ax[3].legend(handles=[Patch(edgecolor="#8a8a85", facecolor=SHADE["beam"][v], label=S.STATE_NAMES["beam"][v]) for v in (0, 1, -1)],
                 ncol=3, loc="upper right", frameon=False)

    bounds = {"acc": (m["t_onset"], m["t_peak_obj_v"]), "dec": (m["t_peak_obj_v"], m["t_first_in_target"]),
              "final": (m["t_first_in_target"], t[-1])}
    for a4 in (ax[4], ax[5]):
        for ph, (a, b) in bounds.items():
            a4.axvspan(a, b, color=SHADE["phase"][ph], lw=0, zorder=0)
    ax[4].plot(t, d["x1"], color=C.P1_COLOR, lw=1.5, label="P1")
    ax[4].plot(t, d["x2"], color=C.P2_COLOR, lw=1.5, ls="--", label="P2")
    for on, col in ((m["onset_P1"], C.P1_COLOR), (m["onset_P2"], C.P2_COLOR)):
        ax[4].axvline(on, color=col, lw=1, ls=":")
    for te, col in ((m["target_entry_P1"], C.P1_COLOR), (m["target_entry_P2"], C.P2_COLOR)):
        if np.isfinite(te):
            ax[4].plot(te, np.interp(te, t, d["x1" if col == C.P1_COLOR else "x2"]), "v", color=col, ms=8)
    ax[4].set_ylabel("reach position x (m)")
    ax[4].set_title(f"(d) Reach: onset (dotted) and final target entry (▼).  "
                    f"init diff {1e3 * m['init_diff']:+.0f} ms, target-entry diff {1e3 * m['target_entry_diff']:+.0f} ms "
                    f"(> 0: P1 first)", loc="left")
    ax[4].legend(handles=[Patch(edgecolor="#8a8a85", facecolor=SHADE["phase"][k], label=M.PHASE_LONG[k].split("\n")[0]) for k in M.PHASES]
                 + [Line2D([], [], color=C.P1_COLOR, label="P1"), Line2D([], [], color=C.P2_COLOR, ls="--", label="P2")],
                 ncol=5, loc="lower right", frameon=False)
    # (e) X reach segmentation as horizontal bars
    for i, (k, col) in enumerate((("r1", C.P1_COLOR), ("r2", C.P2_COLOR))):
        for a, b, v in S.runs(lab[k]):
            if v != 0:
                ax[5].plot([t[a], t[b - 1]], [1 - i] * 2, lw=8, solid_capstyle="butt",
                           color=col, alpha=1.0 if v == 1 else 0.4)
    ax[5].plot(t, d["vx1"] / max(1e-9, np.abs(d[["vx1", "vx2"]].values).max()) * 0.4 + 1, color=C.P1_COLOR, lw=0.8)
    ax[5].plot(t, d["vx2"] / max(1e-9, np.abs(d[["vx1", "vx2"]].values).max()) * 0.4, color=C.P2_COLOR, lw=0.8, ls="--")
    ax[5].set_yticks([1, 0], ["P1", "P2"])
    ax[5].set_ylim(-0.6, 1.6)
    ax[5].set_title("(e) Reach segmentation (HMM on $\\dot x$): solid = forward, light = backward; thin line = $\\dot x$",
                    loc="left")
    ax[5].set_xlabel("time from trial start (s)")
    fig.tight_layout()
    save(fig, path)


# ======================================================================
# generic panels
# ======================================================================
def block_panel(ax, res, col, title=None, ylabel=None, ref=None, early_late=True, scale=1.0):
    """Per block: trial dots + mean ± SD; early (◁) and late (▷) means."""
    for i, b in enumerate(B):
        g = res[res["block"] == b]
        v = scale * g[col].astype(float).values
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.25
        ax.scatter(i + jit, v, s=6, color=BC[b], alpha=0.35, lw=0)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="o", ms=6,
                    color="k", mfc=BC[b], capsize=3, lw=1.2, zorder=5)
        if early_late:
            for part, dxp, mk in (("early", -0.3, "<"), ("late", 0.3, ">")):
                gv = scale * g.loc[g["early_late"] == part, col].astype(float)
                if gv.notna().any():
                    ax.plot(i + dxp, gv.mean(), mk, color=BC[b], mec="k", ms=6, zorder=6)
    if ref is not None:
        ax.axhline(ref, color="grey", lw=0.9, ls="--")
    ax.set_xticks(range(len(B)), B)
    ax.set_title(col if title is None else title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel)


EL_LEGEND = [Line2D([], [], marker="o", color="k", mfc="w", ls="", label="mean ± SD"),
             Line2D([], [], marker="<", color="k", mfc="w", ls="", label=f"first {C.EARLY_N}"),
             Line2D([], [], marker=">", color="k", mfc="w", ls="", label=f"last {C.EARLY_N}")]


def grid_of_panels(res, specs, path, title, ncol=4):
    n = len(specs)
    nr = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nr, ncol, figsize=(3.4 * ncol, 2.7 * nr), squeeze=False)
    for ax, sp in zip(axs.ravel(), specs):
        block_panel(ax, res, **sp)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs[0, 0].legend(handles=EL_LEGEND, loc="best", frameon=False)
    fig.suptitle(title + f"   (C2: n = {int((res['block'] == 'C2').sum())} trials)", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def learning_panel(ax, res, col, title, ref=None, scale=1.0):
    for b in B:
        g = res[res["block"] == b]
        x, y = g["global_idx"].values, scale * g[col].astype(float).values
        ax.plot(x, y, "o", ms=2.5, color=BC[b], alpha=0.6)
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            p = np.polyfit(x[ok], y[ok], 1)
            ax.plot(x, np.polyval(p, x), color=BC[b], lw=2)
    if ref is not None:
        ax.axhline(ref, color="grey", lw=0.9, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("successful trial #")


def learning_grid(res, specs, path, title, ncol=3):
    n = len(specs)
    nr = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nr, ncol, figsize=(4.6 * ncol, 2.6 * nr), squeeze=False)
    for ax, sp in zip(axs.ravel(), specs):
        learning_panel(ax, res, **sp)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs[0, 0].legend(handles=[Patch(color=BC[b], label=C.BLOCK_LONG[b]) for b in B], frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, path)


def curves_by_block(res, curves, rows, path, title, mode="n"):
    """rows: list of (label, [(key, color, ls, legend)], scale)."""
    fig, axs = plt.subplots(len(rows), len(B), figsize=(3.1 * len(B), 2.2 * len(rows)),
                            sharex=True, sharey="row", squeeze=False)
    x = PCT if mode == "n" else M.RT_T
    for j, b in enumerate(B):
        for i, (ylab, series, sc) in enumerate(rows):
            ax = axs[i, j]
            for key, col, ls, lg in series:
                arr = sc * np.asarray(_curves(res, curves, b, f"{mode}_{key}"), float)
                band(ax, x, arr, col, ls, lg, min_frac=0.5 if mode == "rt" else 0)
            if any(k in ("dx", "theta", "vx1", "obj_v", "y1") for k, *_ in series):
                ax.axhline(0, color="grey", lw=0.7, ls="--")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"{C.BLOCK_LONG[b]}\n(n = {int((res['block'] == b).sum())})", color="k")
            if i == len(rows) - 1:
                ax.set_xlabel("% of movement" if mode == "n" else "time from movement onset (s)")
        for i, (_, series, _) in enumerate(rows):
            if len(series) > 1 and j == 0:
                axs[i, 0].legend(frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, path)


# ======================================================================
def fig_interaction(res, curves, path):
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05])
    # (a) stacked bars: all / early / late per block
    ax = fig.add_subplot(gs[0, :3])
    xt, xl, xpos = [], [], 0
    for b in B:
        for part in ("all", "early", "late"):
            g = res[res["block"] == b] if part == "all" else res[(res["block"] == b) & (res["early_late"] == part)]
            bottom = 0
            for k in S.INTERACTION:
                v = g[f"ratio_{k}"].mean()
                ax.bar(xpos, v, bottom=bottom, color=INTER_COLORS[k], width=0.8, edgecolor="white", lw=1)
                bottom += v
            xt.append(xpos)
            xl.append(f"{b}\n{part}" if part == "all" else part)
            xpos += 1
        xpos += 0.6
    ax.set_xticks(xt, xl, fontsize=7)
    ax.set_ylabel("fraction of trial time")
    ax.set_title("(a) Interaction dynamics (paper Fig. 4/5): mean ratio per block, first vs last trials", loc="left")
    ax.legend(handles=[Patch(color=INTER_COLORS[k], label=k) for k in S.INTERACTION], ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    # (b) radar
    ax = fig.add_subplot(gs[0, 3:], polar=True)
    axes_ = ["aligned", "stationary", "opposed", "single"]
    ang = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    for b in B:
        g = res[res["block"] == b]
        vals = [g["ratio_aligned"].mean(), g["ratio_stationary"].mean(),
                1 - g["ratio_opposed"].mean(), 1 - g["ratio_single"].mean()]
        ax.plot(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=BC[b], lw=1.8, label=b)
    ax.set_xticks(ang, ["aligned", "stationary", "1 - opposed", "1 - single"], fontsize=8)
    ax.set_title("(b) Coordination polygon (larger = better)", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False, fontsize=7)
    # (c) composition over real time
    for j, b in enumerate(B):
        ax = fig.add_subplot(gs[1, j])
        arr = np.asarray(_curves(res, curves, b, "rt_inter"), float)
        valid = np.isfinite(arr)
        nvalid = valid.sum(0)
        bottom = np.zeros(arr.shape[1])
        for i, k in enumerate(S.INTERACTION):
            frac = np.where(nvalid >= 0.5 * arr.shape[0], (arr == i).sum(0) / np.maximum(nvalid, 1), np.nan)
            ax.fill_between(M.RT_T, bottom, bottom + np.nan_to_num(frac), color=INTER_COLORS[k], lw=0,
                            where=np.isfinite(frac))
            bottom = bottom + np.nan_to_num(frac)
        ax.set_ylim(0, 1)
        ax.set_title(f"(c) {C.BLOCK_LONG[b]}", fontsize=8)
        ax.set_xlabel("time from movement onset (s)")
        if j == 0:
            ax.set_ylabel("fraction of trials")
    fig.tight_layout()
    save(fig, path)


def fig_temporal(res, path):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    specs = [("init_diff", "initiation diff, onset P2 - P1 (ms)"),
             ("target_entry_diff", "target-entry diff, P2 - P1 (ms)"),
             ("seg_delay_all", "paired X-segment delay, P2 - P1 (ms)")]
    for ax, (col, ttl) in zip(axs[0], specs):
        learning_panel(ax, res, col, ttl + "   > 0: P1 first", ref=0, scale=1e3)
        ax.set_ylabel("ms")
    axs[0, 0].legend(handles=[Patch(color=BC[b], label=C.BLOCK_LONG[b]) for b in B], frameon=False, fontsize=7)
    # fraction P1 first
    ax = axs[1, 0]
    w = 0.25
    for k, (col, lbl) in enumerate((("P1_initiates", "P1 initiates"), ("P1_enters_target", "P1 enters target first"),
                                    ("seg_frac_P1_first", "P1 first in paired segments"))):
        v = [res.loc[res["block"] == b, col].mean() for b in B]
        ax.bar(np.arange(len(B)) + (k - 1) * w, v, w, label=lbl, color=["#2a78d6", "#1baf7a", "#6250d6"][k],
               edgecolor="white")
    ax.axhline(0.5, color="grey", ls="--", lw=0.9)
    ax.set_xticks(range(len(B)), B)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of trials (segments)")
    ax.set_title("Who initiates / completes first", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    # C1a vs C1b around the rest
    ax = axs[1, 1]
    n = C.POST_REST_N
    ca, cb = res[res["block"] == "C1a"], res[res["block"] == "C1b"]
    groups = [(f"C1a\nfirst {n}", ca.head(n)), (f"C1a\nlast {n}", ca.tail(n)),
              (f"C1b first {n}\n(post-rest)", cb.head(n)), (f"C1b\nlast {n}", cb.tail(n))]
    for k, (col, lbl, colr) in enumerate((("init_diff", "initiation", "#2a78d6"),
                                          ("target_entry_diff", "target entry", "#1baf7a"),
                                          ("seg_delay_all", "segment delay", "#6250d6"))):
        mu = [1e3 * g[col].mean() for _, g in groups]
        se = [1e3 * g[col].std() / np.sqrt(max(g[col].notna().sum(), 1)) for _, g in groups]
        ax.errorbar(np.arange(4) + (k - 1) * 0.12, mu, yerr=se, fmt="o-", color=colr, capsize=3, label=lbl)
    ax.axhline(0, color="grey", ls="--", lw=0.9)
    ax.axvspan(1.5, 1.55, color="k", alpha=0.6)
    ax.text(1.53, ax.get_ylim()[1], " 60 s rest", va="top", fontsize=7)
    ax.set_xticks(range(4), [g[0] for g in groups], fontsize=7)
    ax.set_ylabel("ms (mean ± SEM), > 0: P1 first")
    ax.set_title("C1 before vs after the rest", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    # absolute delays
    block_panel(axs[1, 2], res, "seg_absdelay_all", "absolute paired X-segment delay (ms)\n(paper's 'absolute delay', on the reach)",
                scale=1e3)
    fig.suptitle("Temporal coordination along the reach (X)", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def fig_phases(res, curves, path):
    rows = [("mean_dx_mm", "mean dx = x1 - x2 (mm)\n> 0: P1 ahead", 0),
            ("frac_P1_ahead", "fraction of time P1 ahead", 0.5),
            ("lag_vx", "xcorr lag on $\\dot x$ (ms)\n> 0: P1 leads (r ≥ %.1f only)" % C.XCORR_MIN_R, 0),
            ("aligned", "aligned ratio (vertical)", None),
            ("corr_y_P1_mm", "corrective mvmt P1, y (mm)", None),
            ("corr_y_P2_mm", "corrective mvmt P2, y (mm)", None)]
    fig, axs = plt.subplots(len(rows), 3, figsize=(12, 2.3 * len(rows)), sharey="row")
    for j, ph in enumerate(M.PHASES):
        for i, (k, lab, ref) in enumerate(rows):
            sc = 1e3 if k == "lag_vx" else 1.0
            block_panel(axs[i, j], res, f"{ph}_{k}", title=(M.PHASE_LONG[ph].replace("\n", " ") if i == 0 else ""),
                        ref=ref, scale=sc)
            if j == 0:
                axs[i, j].set_ylabel(lab, fontsize=8)
    axs[0, 0].legend(handles=EL_LEGEND, frameon=False, fontsize=7)
    fig.suptitle("Lead-lag by movement phase", fontsize=11)
    fig.tight_layout()
    save(fig, path)
    # TLCC heatmaps
    fig, axs = plt.subplots(1, len(B), figsize=(3.2 * len(B), 3.2), sharey=True)
    for ax, b in zip(axs, B):
        mats = [m for m in _curves(res, curves, b, "tlcc") if m is not None]
        if not mats:
            continue
        mat = np.nanmean(np.stack(mats), 0)
        lim = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[0, 100, 1e3 * M.TLCC_LAGS[0], 1e3 * M.TLCC_LAGS[-1]])
        pk = M.TLCC_LAGS[np.nanargmax(np.nan_to_num(mat, nan=-9), 0)]
        ax.plot((np.arange(mat.shape[1]) + 0.5) / mat.shape[1] * 100, 1e3 * pk, "k.-", lw=1)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{C.BLOCK_LONG[b]}", fontsize=8)
        ax.set_xlabel("% of movement")
        fig.colorbar(im, ax=ax, fraction=0.05)
    axs[0].set_ylabel("lag τ (ms), > 0: P1 leads")
    fig.suptitle("Windowed cross-correlation of $\\dot x_1$, $\\dot x_2$ along the movement (mean r; black = peak lag)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, path.replace(".png", "_tlcc.png"))


def fig_variability(var, fail, path):
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    cols = [("var_x1_mm", "x1 (mm)"), ("var_x2_mm", "x2 (mm)"), ("var_dx_mm", "dx (mm)"), ("var_theta_deg", "θ (deg)"),
            ("var_y1_mm", "y1 (mm)"), ("var_y2_mm", "y2 (mm)"), ("endpoint_sd_P1_mm", "endpoint SD P1 (mm)")]
    for ax, (c, lbl) in zip(axs.ravel(), cols):
        for k, part in enumerate(("all", "early", "late")):
            v = [var.loc[(var["block"] == b) & (var["part"] == part), c].mean() for b in B]
            ax.bar(np.arange(len(B)) + (k - 1) * 0.27, v, 0.27, color=[BC[b] for b in B],
                   alpha=[1.0, 0.45, 0.75][k], edgecolor="white", hatch=["", "//", ".."][k], label=part)
        ax.set_xticks(range(len(B)), B)
        ax.set_title("across-trial SD: " + lbl if not c.startswith("endpoint") else lbl, fontsize=9)
    axs[0, 0].legend(frameon=False, fontsize=7, title="trials", title_fontsize=7)
    ax = axs[1, 3]
    ax.bar(range(len(B)), 100 * fail["failure_rate"].values, color=[BC[b] for b in B], edgecolor="white")
    ax.set_xticks(range(len(B)), B)
    ax.set_title("failure rate (%)", fontsize=9)
    fig.suptitle("Movement variability (SD across trials of the movement-normalised curves)", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def fig_hmm(seg, data, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
    sigs = {"vert": ("vy1", "vy2"), "beam": ("theta_dot",), "reach": ("vx1", "vx2")}
    for ax, (k, sg) in zip(axs, seg.items()):
        v = np.concatenate([d[c].values for d in data.values() for c in sigs[k]])
        ax.hist(v, bins=200, density=True, color="#c3c2b7")
        if sg.method == "hmm":
            xs = np.linspace(*np.percentile(v, [0.5, 99.5]), 400)
            w = np.bincount(sg.map[sg.model.predict((v[::25] / sg.scale)[:, None])] + 1, minlength=3) / len(v[::25])
            for i, (mu, sd) in enumerate(zip(sg.info["means"], sg.info["sds"])):
                ax.plot(xs, w[i] * np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                        lw=1.8, color=["#eb6834", "#3d3d3a", "#1baf7a"][i], label=S.STATE_NAMES[k][i - 1])
        ax.set_xlim(*np.percentile(v, [0.5, 99.5]))
        ax.set_title(f"HMM states: {k} ({'/'.join(sigs[k])})", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, path)


# ======================================================================
def make_all(res, curves, data, inters, var, fail, seg, fdir):
    # F0 examples
    if C.EXAMPLE_TRIALS == "auto":
        ex = []
        for b in B:
            g = res[res["block"] == b]
            ex += [(g.iloc[0].block_num, g.iloc[0].trial), (g.iloc[-1].block_num, g.iloc[-1].trial)]
    else:
        ex = C.EXAMPLE_TRIALS
    exdir = os.path.join(fdir, "F0_example_trials")
    os.makedirs(exdir, exist_ok=True)
    for key in ex:
        r = res[(res["block_num"] == key[0]) & (res["trial"] == key[1])]
        if r.empty:
            continue
        r = r.iloc[0]
        lab, inter = inters[key]
        fig_example_trial(data[key], lab, inter, r, f"{C.BLOCK_LONG[r.block]}, trial {r.trial}",
                          os.path.join(exdir, f"F0_{r.block}_trial{int(r.trial):02d}.png"))

    traj_rows = [("reach x (m)", [("x1", C.P1_COLOR, "-", "P1"), ("x2", C.P2_COLOR, "--", "P2")], 1),
                 ("dx = x1 - x2 (mm)", [("dx", C.OBJ_COLOR, "-", None)], 1e3),
                 ("height y (mm)", [("y1", C.P1_COLOR, "-", "P1"), ("y2", C.P2_COLOR, "--", "P2")], 1e3),
                 ("beam tilt θ (deg)", [("theta", C.OBJ_COLOR, "-", None)], 1),
                 ("$\\dot x$ (m/s)", [("vx1", C.P1_COLOR, "-", "P1"), ("vx2", C.P2_COLOR, "--", "P2")], 1)]
    curves_by_block(res, curves, traj_rows, os.path.join(fdir, "F1_trajectories_normalised.png"),
                    "Trajectories, movement-normalised (mean ± SD across successful trials)", "n")
    curves_by_block(res, curves, traj_rows, os.path.join(fdir, "F1b_trajectories_realtime.png"),
                    "Trajectories in real time from movement onset (drawn while ≥ 50 % of trials are running)", "rt")

    grid_of_panels(res, [
        dict(col="mean_abs_dx_mm", title="mean |dx| (mm)"), dict(col="mean_dx_mm", title="mean dx (mm), > 0 P1 ahead", ref=0),
        dict(col="peak_abs_dx_mm", title="peak |dx| (mm)"), dict(col="mean_abs_dy_mm", title="mean |y1 - y2| (mm)"),
        dict(col="corr_y_P1_mm", title="corrective mvmt P1, vertical (mm)"),
        dict(col="corr_y_P2_mm", title="corrective mvmt P2, vertical (mm)"),
        dict(col="CR_y_P1", title="corrective ratio P1 (vertical)", ref=0.5),
        dict(col="CR_x_P1", title="corrective ratio P1 (reach x)", ref=0.5),
        dict(col="corr_x_P1_mm", title="corrective mvmt P1, reach x (mm)"),
        dict(col="corr_x_P2_mm", title="corrective mvmt P2, reach x (mm)"),
        dict(col="n_backward_P1", title="# backward X segments P1"),
        dict(col="n_backward_P2", title="# backward X segments P2"),
    ], os.path.join(fdir, "F2_position_difference_corrective.png"), "Position difference and corrective movements")
    fig_variability(var, fail, os.path.join(fdir, "F2b_variability.png"))

    grid_of_panels(res, [
        dict(col="sparc_P1", title="SPARC P1 (closer to 0 = smoother)"), dict(col="sparc_P2", title="SPARC P2"),
        dict(col="sparc_obj", title="SPARC beam centre"), dict(col="movement_time", title="movement time (s)"),
        dict(col="n_submov_P1", title="# submovements P1"), dict(col="n_submov_P2", title="# submovements P2"),
        dict(col="peak_obj_v", title="peak beam-centre velocity (m/s)"), dict(col="trial_time", title="trial time (s)"),
    ], os.path.join(fdir, "F3_smoothness.png"), "Movement smoothness")

    grid_of_panels(res, [
        dict(col="max_abs_theta", title="max |θ| (deg)"), dict(col="rms_theta", title="RMS θ (deg)"),
        dict(col="mean_theta", title="mean θ (deg), > 0 P1 end up", ref=0), dict(col="end_theta", title="final θ (deg)", ref=0),
        dict(col="r_theta_objv", title="r(θ, beam-centre velocity)", ref=0),
        dict(col="r_theta_dx", title="r(θ, dx)", ref=0),
        dict(col="r_absthetadot_speed", title="r(|dθ/dt|, speed)", ref=0),
        dict(col="lag_objv_theta", title="lag velocity→θ (s), > 0 velocity leads", ref=0),
        dict(col="frac_time_rotating", title="fraction of time beam rotating"),
        dict(col="n_rot_segments", title="# beam rotation segments"),
        dict(col="mean_rot_dur", title="mean rotation segment duration (s)"),
        dict(col="unilateral_ratio_P1", title="unilateral manipulation ratio P1", ref=0.5),
    ], os.path.join(fdir, "F4_beam_tilt.png"), "Beam tilt and its relation to the reach")

    fig_interaction(res, curves, os.path.join(fdir, "F5_interaction_dynamics.png"))
    fig_temporal(res, os.path.join(fdir, "F6_temporal_initiation.png"))
    fig_phases(res, curves, os.path.join(fdir, "F7_phase_leadlag.png"))

    learning_grid(res, [
        dict(col="trial_time", title="trial time (s)"), dict(col="mean_abs_dx_mm", title="mean |dx| (mm)"),
        dict(col="max_abs_theta", title="max |θ| (deg)"), dict(col="sparc_obj", title="SPARC beam centre"),
        dict(col="CR_y_P1", title="corrective ratio P1 (vertical)", ref=0.5),
        dict(col="ratio_aligned", title="aligned ratio"), dict(col="ratio_opposed", title="opposed ratio"),
        dict(col="ratio_single", title="single ratio"),
        dict(col="unilateral_ratio_P1", title="unilateral ratio P1", ref=0.5),
        dict(col="init_diff", title="initiation diff (ms), > 0 P1 first", ref=0, scale=1e3),
        dict(col="target_entry_diff", title="target-entry diff (ms), > 0 P1 first", ref=0, scale=1e3),
        dict(col="seg_delay_all", title="paired segment delay (ms), > 0 P1 first", ref=0, scale=1e3),
    ], os.path.join(fdir, "F8_learning_curves.png"), "Learning curves (successful trials; line = linear fit per block)")
    fig_hmm(seg, data, os.path.join(fdir, "F9_hmm_states.png"))
