"""
Across-dyad summary.  Protocols are kept separate:

    Summary/ProtocolA/  (1 dyad: Baseline - C1 - rest - C2)
    Summary/ProtocolB/  (4 dyads: Baseline - C1 - rest - Baseline - C2)
    Summary/Pooled/     Baseline 1 + C1 only (identical history in both
                        protocols), plus an all-phase plot with protocol marked
    Summary/all_trials.csv, Summary/all_phase_summary.csv

Statistics follow the dyad as the unit of analysis: every dyad is first
averaged over its trials, then the group mean +/- SEM is taken ACROSS DYADS
(pooling all trials of all dyads would treat ~60 correlated trials per dyad
as independent samples).

Run:  python3 overlay_analysis.py      (computes everything from raw data;
                                        run_analysis.py is not required first)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings as _w
import numpy as np
import pandas as pd

# all-NaN slices (e.g. a metric undefined in one phase) are expected
_w.filterwarnings("ignore", category=RuntimeWarning)

import config as C
import plot_utils as P
from plot_utils import plt
from run_analysis import process_pilot, _phase_recs, LF_PANELS

VAR_METRICS = [
    ("var_x1", "Position SD P1 (m)"), ("var_x2", "Position SD P2 (m)"),
    ("var_tilt", "Tilt SD (deg)"), ("var_v1", "Velocity SD P1 (m/s)"),
    ("var_v2", "Velocity SD P2 (m/s)"), ("var_dx", "Position-difference SD (m)"),
    ("endpoint_sd_P1", "Endpoint SD P1 (m)"), ("endpoint_sd_P2", "Endpoint SD P2 (m)"),
    ("cv_movement_time", "CV movement time"),
]
PROTO_MARKERS = {"A": "s", "B": "o"}


# ======================================================================
# HELPERS
# ======================================================================
def dyad_curve(res, ph, key):
    R = _phase_recs(res["df"], res["recs"], ph)
    if not R:
        return None
    return np.nanmean([r[key] for r in R], axis=0)


def curves_figure(group, phases, rows, title, path):
    """rows: list of (ylabel, [(key, linestyle), ...]). Thin = dyad mean,
    thick black = group mean +/- SEM across dyads."""
    fig, axes = plt.subplots(len(rows), len(phases), figsize=(4 * len(phases), 3.2 * len(rows)),
                             sharey="row", squeeze=False)
    for j, ph in enumerate(phases):
        for i, (ylab, keys) in enumerate(rows):
            ax = axes[i, j]
            for key, ls in keys:
                allc = []
                for res in group:
                    c = dyad_curve(res, ph, key)
                    if c is None:
                        continue
                    allc.append(c)
                    ax.plot(P.PCT, c, color=res["color"], ls=ls, lw=1.1, alpha=0.8,
                            label=res["cfg"]["id"] if (i == 0 and j == 0 and ls == C.P1_LS) else None)
                if len(allc) > 1:
                    P.band(ax, P.PCT, allc, "black", ls, lw=2.4, alpha=0.15, sem=True,
                           label=("group mean +/- SEM" if (i == 0 and j == 0 and ls == C.P1_LS) else None))
            ax.axhline(0, color="grey", lw=0.6)
            if i == 0:
                ax.set_title(P.phase_label(ph))
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(rows) - 1:
                ax.set_xlabel("Movement completed (%)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title + "   (P1 solid, P2 dashed)", fontsize=11)
    P.save(fig, path)


def phase_means_figure(group, phases, metrics, title, path, source="phase_df", show_mean=True):
    """Each dyad = one line across phases; black = mean +/- SEM across dyads."""
    nc = 4
    nr = int(np.ceil(len(metrics) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.3 * nr), squeeze=False)
    xs = np.arange(len(phases))
    for ax, m in zip(axes.ravel(), metrics):
        col, lab = m[0], m[1]
        zero = m[2] if len(m) > 2 else False
        note = m[3] if len(m) > 3 else None
        mat = []
        for res in group:
            pdf = res[source].set_index("phase")
            y = np.array([pdf.loc[ph, col] if (ph in pdf.index and col in pdf) else np.nan
                          for ph in phases], float)
            mat.append(y)
            ax.plot(xs, y, marker=PROTO_MARKERS[res["cfg"]["protocol"]], ms=5, lw=1,
                    color=res["color"], alpha=0.85, label=res["cfg"]["id"])
        mat = np.array(mat)
        n = np.sum(np.isfinite(mat), axis=0)
        if show_mean and len(group) > 1:
            mu = np.nanmean(mat, axis=0)
            se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            ax.errorbar(xs + 0.08, mu, se, color="k", lw=2.2, capsize=4, marker="D", ms=5,
                        label="mean +/- SEM", zorder=5)
        P._ref_line(ax, zero)
        if note:
            ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", fontsize=7, color="grey")
        ax.set_xticks(xs)
        ax.set_xticklabels(phases, rotation=20, fontsize=7.5)
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlim(-0.4, len(phases) - 0.6)
    axes[0, 0].legend(fontsize=6.5)
    for a in axes.ravel()[len(metrics):]:
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    P.save(fig, path)


def across_trials_figure(group, metrics, title, path):
    """Metric vs chronological trial; group mean per trial index (+/- SEM).
    Only meaningful within one protocol (same trial sequence)."""
    nc = 4
    nr = int(np.ceil(len(metrics) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3 * nr), squeeze=False)
    ref = max(group, key=lambda r: len(r["df"]))["df"]
    for ax, m in zip(axes.ravel(), metrics):
        col, lab, zero = m[0], m[1], m[2]
        P.shade_phases(ax, ref)
        mat = {}
        for res in group:
            d = res["df"]
            if col not in d:
                continue
            ax.plot(d["global_trial_idx"], d[col], color=res["color"], lw=0.7, alpha=0.5)
            for i, v in zip(d["global_trial_idx"], d[col].astype(float)):
                mat.setdefault(i, []).append(v)
        if len(group) > 1 and mat:
            idx = np.array(sorted(mat))
            mu = np.array([np.nanmean(mat[i]) for i in idx])
            se = np.array([np.nanstd(mat[i], ddof=1) / np.sqrt(np.sum(np.isfinite(mat[i])))
                           if np.sum(np.isfinite(mat[i])) > 1 else np.nan for i in idx])
            ax.plot(idx, mu, "k", lw=1.8)
            ax.fill_between(idx, mu - se, mu + se, color="k", alpha=0.15, lw=0)
        P._ref_line(ax, zero)
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlabel("Trial (chronological)")
    for a in axes.ravel()[len(metrics):]:
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    P.save(fig, path)


def early_late_figure(group, phases, metrics, title, path):
    nc = 4
    nr = int(np.ceil(len(metrics) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.2 * nr), squeeze=False)
    w = 0.38
    for ax, m in zip(axes.ravel(), metrics):
        col, lab, zero = m[0], m[1], m[2]
        for j, ph in enumerate(phases):
            for k, part in enumerate(("early", "late")):
                vals = []
                for res in group:
                    e = res["el_df"]
                    v = e.loc[(e.phase == ph) & (e.part == part), col] if col in e else []
                    if len(v):
                        vals.append(float(v.values[0]))
                        ax.scatter(j + (k - 0.5) * w, v.values[0], color=res["color"], s=12, zorder=4)
                vals = np.array(vals, float)
                if np.isfinite(vals).any():
                    ax.bar(j + (k - 0.5) * w, np.nanmean(vals), w * 0.9, color=P.phase_color(ph),
                           alpha=0.35 if part == "early" else 0.85, edgecolor="k", lw=0.5,
                           label=part if j == 0 else None)
        P._ref_line(ax, zero)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, rotation=20, fontsize=7.5)
        ax.set_title(lab, fontsize=8.5)
    axes[0, 0].legend(fontsize=7)
    for a in axes.ravel()[len(metrics):]:
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    P.save(fig, path)


def group_table(group, phases):
    """Mean, SEM and n across dyads for every phase-level metric."""
    pdf = pd.concat([r["phase_df"] for r in group], ignore_index=True)
    pdf = pdf[pdf["phase"].isin(phases)]
    num = pdf.select_dtypes("number").columns
    g = pdf.groupby("phase")[list(num)]
    tab = pd.concat({"mean": g.mean(), "sem": g.sem(), "n_dyads": g.count()}, axis=1)
    return tab.reindex(phases)


# ======================================================================
# GROUP OUTPUT
# ======================================================================
def make_group(group, phases, name, outdir, across_trials=True):
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    tag = f"{name}  (n = {len(group)} dyad{'s' if len(group) != 1 else ''})"

    curves_figure(group, phases, [("Hand position x (m)", [("x1", C.P1_LS), ("x2", C.P2_LS)])],
                  f"O1. Trajectories - {tag}", f"{fd}/O1_trajectories.png")
    curves_figure(group, phases, [("Velocity x (m/s)", [("v1", C.P1_LS), ("v2", C.P2_LS)]),
                                  ("v1 - v2 (m/s)", [("dv", "-")])],
                  f"O2. Velocity - {tag}", f"{fd}/O2_velocity.png")
    curves_figure(group, phases, [("Beam tilt (deg)", [("tilt", "-")]),
                                  ("x1 - x2 (m)", [("dx", "-")])],
                  f"O3. Beam tilt and position difference - {tag}", f"{fd}/O3_tilt_posdiff.png")
    curves_figure(group, phases, [("|F| logged (N)", [("F1", C.P1_LS), ("F2", C.P2_LS)]),
                                  (f"m*a_x (N, m_eff={C.M_EFF_KG:g})", [("Fk1", C.P1_LS), ("Fk2", C.P2_LS)])],
                  f"O4. Forces (logged and kinematic) - {tag}", f"{fd}/O4_force_curves.png")
    phase_means_figure(group, phases, P.KEY_METRICS,
                       f"O5. Key metrics per phase - {tag}", f"{fd}/O5_phase_means.png")
    phase_means_figure(group, phases, [(c, t, z, "> 0 : P1 leads" if z is True else None)
                                       for c, _, t, z in LF_PANELS],
                       f"O6. Leader-follower metrics per phase (+ = P1 leads) - {tag}",
                       f"{fd}/O6_leader_follower.png")
    phase_means_figure(group, phases, VAR_METRICS,
                       f"O7. Trial-to-trial variability per phase - {tag}", f"{fd}/O7_variability.png")
    early_late_figure(group, phases, P.KEY_METRICS,
                      f"O8. First {C.EARLY_N} vs last {C.LATE_N} trials (bar = mean of dyads, dots = dyads) - {tag}",
                      f"{fd}/O8_early_vs_late.png")
    if across_trials:
        across_trials_figure(group, P.KEY_METRICS,
                             f"O9. Across trials (thin = dyad, black = mean +/- SEM) - {tag}",
                             f"{fd}/O9_across_trials.png")
    group_table(group, phases).to_csv(f"{outdir}/group_phase_means.csv")
    pd.concat([r["phase_df"] for r in group]).to_csv(f"{outdir}/dyad_phase_summary.csv", index=False)
    pd.concat([r["el_df"] for r in group]).to_csv(f"{outdir}/dyad_early_late_summary.csv", index=False)
    print(f"  -> {outdir}/")


def main():
    results = []
    for i, pcfg in enumerate(C.PILOTS):
        if not os.path.isdir(pcfg["folder"]):
            print(f"SKIP {pcfg['id']}: folder not found: {pcfg['folder']}")
            continue
        print(f"\n--- {pcfg['id']} ({pcfg['protocol']}) ---")
        try:
            res = process_pilot(pcfg)
        except Exception as e:
            print(f"ERROR in {pcfg['id']}: {e!r}")
            continue
        res["color"] = C.PILOT_COLORS[i % len(C.PILOT_COLORS)]
        results.append(res)
    if not results:
        print("No data loaded - check config.PILOTS.")
        return

    os.makedirs(C.SUMMARY_ROOT, exist_ok=True)
    pd.concat([r["df"].assign(pilot=r["cfg"]["id"], protocol=r["cfg"]["protocol"]) for r in results],
              ignore_index=True).to_csv(f"{C.SUMMARY_ROOT}/all_trials.csv", index=False)
    pd.concat([r["phase_df"] for r in results], ignore_index=True).to_csv(
        f"{C.SUMMARY_ROOT}/all_phase_summary.csv", index=False)

    print("\nBuilding group figures")
    for proto, spec in C.PROTOCOLS.items():
        group = [r for r in results if r["cfg"]["protocol"] == proto]
        if group:
            make_group(group, spec["phases"], spec["name"], os.path.join(C.SUMMARY_ROOT, f"Protocol{proto}"))

    # pooled: only phases with identical history in both protocols
    pooled_dir = os.path.join(C.SUMMARY_ROOT, "Pooled")
    make_group(results, C.POOLABLE_PHASES, "All dyads, common phases only", pooled_dir,
               across_trials=False)
    # all phases, protocols marked (squares = A, circles = B); group mean
    # omitted because C2 has a different history in the two protocols
    all_ph = ["Baseline 1", "C1", "Baseline 2", "C2"]
    fd = os.path.join(pooled_dir, "figures")
    phase_means_figure(results, all_ph, P.KEY_METRICS,
                       "O10. All dyads, all phases (square = protocol A, circle = protocol B; "
                       "no group mean: C2 follows a washout only in protocol B)",
                       f"{fd}/O10_all_phases_by_protocol.png", show_mean=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
