"""
Quick, low-memory check of the raw files of every dyad in config.PILOTS.
Reads only the 'Time' column of the logs.

    python3 diagnose.py
"""
import os
import numpy as np
import pandas as pd
import config as C

for pc in C.PILOTS:
    f = pc["folder"]
    print(f"\n=== {pc['id']}  ({f})")
    if not os.path.isdir(f):
        print("   folder not found")
        continue
    try:
        tr = pd.read_csv(os.path.join(f, "trial.csv"), encoding="utf-8-sig")
        tr.columns = [c.strip() for c in tr.columns]
        n_trials = tr.groupby(["block num", "trial num"]).ngroups
        print(f"   trial.csv: {len(tr)} rows, {n_trials} trials, conditions in order: "
              f"{list(dict.fromkeys(tr['condition'].astype(str).str.strip()))}")
        ev = tr["phase"].astype(str).str.strip().value_counts().to_dict()
        print(f"   events: {ev}")
    except Exception as e:
        print(f"   trial.csv: ERROR {e!r}")
    for name in ("primarySpringCouplingLog.csv", "secondarySpringCouplingLog.csv"):
        path = os.path.join(f, name)
        if not os.path.exists(path):
            print(f"   {name}: MISSING")
            continue
        size = os.path.getsize(path) / 1e6
        hdr = [c.strip() for c in pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns]
        t = pd.read_csv(path, encoding="utf-8-sig",
                        usecols=lambda c: c.strip() == "Time")["Time"].to_numpy(float)
        dt = np.diff(np.sort(t))
        n_win = int((dt > 1.0).sum()) + 1
        print(f"   {name}: {size:.0f} MB, {len(t):,} rows, {len(hdr)} columns")
        print(f"      Time {t.min():.2f} .. {t.max():.2f} (span {t.max() - t.min():.0f}), "
              f"median dt {np.median(dt) * 1000:.2f} ms, logging windows (gap > 1 s): {n_win}")
        if np.median(dt) > 0.5:
            print("      !! median dt > 0.5 -> Time is probably not in seconds")
print()
