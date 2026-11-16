"""
Run the full analysis:  python3 analysis_v2/run_v2.py
Outputs -> analysis_v2/Results/<dataset id>/
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as C           # noqa: E402
import io_v2 as IO              # noqa: E402
import segmentation as S        # noqa: E402
import metrics_v2 as M          # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def curve_var(arrs):
    arr = np.asarray(arrs, float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.nan
    return float(np.nanmean(np.nanstd(arr, axis=0, ddof=1)))


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan


def run_dataset(ds):
    out = os.path.join(C.OUT_ROOT, ds["id"])
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    qc = [f"Dataset {ds['id']}  ({ds['folder']})"]

    trial, p, s = IO.load(ds["folder"])
    tt_all = IO.build_trial_table(trial)
    sy = IO.sync(tt_all, p, s)
    qc.append(f"Clock sync: offset {sy['offset']:.4f} s, {sy['n_matched']}/{sy['n_trials']} trial "
              f"starts matched to {sy['n_windows']} log windows, spread {sy['spread']:.4f} s")

    main = tt_all[tt_all["block"].notna()]
    fail = (main.groupby("block")["result"]
            .agg(n_total="size", n_success=lambda r: (r == "success").sum()).reindex(C.BLOCK_ORDER))
    fail["failure_rate"] = 1 - fail["n_success"] / fail["n_total"]
    tt = IO.select_trials(tt_all)

    # ---------- resample ----------
    data, kept, qrows = {}, [], []
    for _, row in tt.iterrows():
        d, q = IO.resample_trial(row, p, s)
        q.update(block=row["block"], trial=row["trial"])
        qrows.append(q)
        if d is not None:
            data[(row["block_num"], row["trial"])] = d
            kept.append(row)
    tt = pd.DataFrame(kept).reset_index(drop=True)
    qdf = pd.DataFrame(qrows)
    qc.append(f"Successful main trials: {len(qrows)}, segmented: {len(tt)} "
              f"(dropped {len(qrows) - len(tt)} with log coverage < {C.MIN_COVERAGE:.0%})")
    qc.append(f"Coverage median {qdf['coverage'].median():.3f}, min {qdf['coverage'].min():.3f}; "
              f"max log gap {qdf['max_gap'].max():.3f} s")
    qc.append(f"Partner height, Primary 'OtherPosY' vs Secondary 'MyPosY': RMS diff median "
              f"{qdf['y2_rms_diff_mm'].median():.3f} mm;  partner X: {qdf['x2_rms_diff_mm'].median():.3f} mm")

    # ---------- segmentation models ----------
    ds_ = list(data.values())
    seg = {
        "vert": S.Segmenter("vert").fit([d["vy1"].values for d in ds_] + [d["vy2"].values for d in ds_]),
        "beam": S.Segmenter("beam").fit([d["theta_dot"].values for d in ds_]),
        "reach": S.Segmenter("reach").fit([d["vx1"].values for d in ds_] + [d["vx2"].values for d in ds_]),
    }
    for k, sg in seg.items():
        qc.append(f"Segmentation [{k}]: {json.dumps(sg.info)}")

    # ---------- metrics ----------
    rows, curves, segrows, inters = [], {}, [], {}
    for _, row in tt.iterrows():
        key = (row["block_num"], row["trial"])
        d = data[key]
        lab = dict(v1=seg["vert"].decode(d["vy1"].values), v2=seg["vert"].decode(d["vy2"].values),
                   beam=seg["beam"].decode(d["theta_dot"].values),
                   r1=seg["reach"].decode(d["vx1"].values), r2=seg["reach"].decode(d["vx2"].values))
        m, cv, sr, inter = M.trial_metrics(d, row, lab)
        base = row[["block", "block_num", "trial", "trial_in_block", "n_in_block",
                    "early_late", "global_idx", "trial_duration",
                    "target_entry_P1", "target_entry_P2"]].to_dict()
        rows.append({**base, **m})
        curves[key] = cv
        inters[key] = (lab, inter)
        segrows += [{"block": row["block"], "trial": row["trial"], **r} for r in sr]
    res = pd.DataFrame(rows)
    segdf = pd.DataFrame(segrows)

    # sanity checks
    rs = res[[f"ratio_{k}" for k in S.INTERACTION]].sum(axis=1)
    assert np.allclose(rs, 1.0), "interaction ratios do not sum to 1"
    bad = res[(res["onset_P1"] >= res["offset_P1"]) | (res["onset_P2"] >= res["offset_P2"])]
    qc.append(f"Trials with onset >= offset: {len(bad)};  onset after 1.5 s: "
              f"P1 {(res['onset_P1'] > 1.5).sum()}, P2 {(res['onset_P2'] > 1.5).sum()}")

    # ---------- summaries ----------
    metric_cols = [c for c in res.columns if c not in
                   ("block", "block_num", "trial", "trial_in_block", "n_in_block",
                    "early_late", "global_idx") and pd.api.types.is_numeric_dtype(res[c])]
    bs = []
    for b in C.BLOCK_ORDER:
        g = res[res["block"] == b]
        for c in metric_cols:
            bs.append(dict(block=b, metric=c, n=int(g[c].notna().sum()), mean=g[c].mean(),
                           sd=g[c].std(), median=g[c].median(),
                           slope_per_trial=slope(g["trial_in_block"], g[c])))
    bsum = pd.DataFrame(bs)
    el = (res[res["early_late"] != "mid"].groupby(["block", "early_late"])[metric_cols]
          .agg(["mean", "std"]))
    # variability (across-trial SD of normalised curves) per block and early/late
    var_rows = []
    for b in C.BLOCK_ORDER:
        for part in ("all", "early", "late"):
            g = res[(res["block"] == b) & ((res["early_late"] == part) if part != "all" else True)]
            keys = [(r.block_num, r.trial) for r in g.itertuples()]
            vr = dict(block=b, part=part, n=len(keys))
            for k in ("x1", "x2", "dx", "y1", "y2", "theta"):
                sc = 1e3 if k != "theta" else 1.0
                vr[f"var_{k}" + ("_mm" if k != "theta" else "_deg")] = sc * curve_var(
                    [curves[q]["n_" + k] for q in keys])
            vr["endpoint_sd_P1_mm"] = 1e3 * g["end_x_P1"].std()
            vr["endpoint_sd_P2_mm"] = 1e3 * g["end_x_P2"].std()
            vr["cv_movement_time"] = g["movement_time"].std() / g["movement_time"].mean()
            vr["cv_trial_time"] = g["trial_time"].std() / g["trial_time"].mean()
            var_rows.append(vr)
    var = pd.DataFrame(var_rows)
    # C1: end of block A vs right after the rest vs end of block B
    pr_cols = ["init_diff", "complete_diff", "target_entry_diff", "hmm_init_diff", "seg_delay_all",
               "seg_absdelay_all", "P1_initiates", "onset_P1", "onset_P2", "movement_time",
               "trial_time", "max_abs_theta", "ratio_aligned", "ratio_opposed"]
    n = C.POST_REST_N
    ca, cb = res[res["block"] == "C1a"], res[res["block"] == "C1b"]
    pr = pd.DataFrame({
        f"C1a first {n}": ca.head(n)[pr_cols].mean(), f"C1a last {n}": ca.tail(n)[pr_cols].mean(),
        f"C1b first {n} (post-rest)": cb.head(n)[pr_cols].mean(), f"C1b last {n}": cb.tail(n)[pr_cols].mean()})

    res.to_csv(os.path.join(out, "trial_summary.csv"), index=False)
    bsum.to_csv(os.path.join(out, "block_summary.csv"), index=False)
    bsum.pivot(index="metric", columns="block", values="mean")[C.BLOCK_ORDER].to_csv(
        os.path.join(out, "block_means_wide.csv"))
    el.T.to_csv(os.path.join(out, "early_late.csv"))
    var.to_csv(os.path.join(out, "variability.csv"), index=False)
    pr.to_csv(os.path.join(out, "C1_post_rest.csv"))
    fail.to_csv(os.path.join(out, "failure_rate.csv"))
    segdf.to_csv(os.path.join(out, "segments.csv"), index=False)
    with open(os.path.join(out, "qc_report.txt"), "w") as f:
        f.write("\n".join(qc) + "\n\nFailure rate per block:\n" + fail.to_string() + "\n")
    print("\n".join(qc))

    import plots_v2 as P
    P.make_all(res, curves, data, inters, var, fail, seg, os.path.join(out, "figures"))
    return res


if __name__ == "__main__":
    only = sys.argv[1:]
    for ds in C.DATASETS:
        if not only or ds["id"] in only:
            run_dataset(ds)
