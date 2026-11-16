"""
Movement segmentation (Liu et al. 2025, Methods - Movement segmentation).

A 3-state Gaussian HMM is fitted on a velocity signal pooled over all trials
of the session; states are ordered by their mean velocity and labelled
-1 (down / cw / backward), 0 (stationary), +1 (up / ccw / forward).
Decoding uses Viterbi; runs shorter than MIN_SEG_S are merged into the
preceding segment.
"""
import warnings

import numpy as np
import pandas as pd

import config_v2 as C

STATE_NAMES = {
    "vert": {-1: "down", 0: "stationary", 1: "up"},
    "beam": {-1: "cw (P2 end up)", 0: "stationary", 1: "ccw (P1 end up)"},
    "reach": {-1: "backward", 0: "stationary", 1: "forward"},
}
INTERACTION = ["stationary", "single", "aligned", "opposed"]

_STEP = int(round(C.FS / C.HMM_FS))


class Segmenter:
    """Fit once on a list of velocity arrays (FS Hz), then decode trials."""

    def __init__(self, kind, method=C.SEG_METHOD):
        self.kind, self.method = kind, method
        self.info = {}

    def fit(self, seqs):
        ds = [np.asarray(v, float)[::_STEP] for v in seqs if len(v) > 5 * _STEP]
        allv = np.concatenate(ds)
        self.scale = float(np.std(allv)) or 1.0
        mad = 1.4826 * np.median(np.abs(allv - np.median(allv)))
        self.robust_sd = float(mad) if mad > 0 else self.scale
        if self.method == "threshold":
            self.info = dict(method="threshold", robust_sd=self.robust_sd)
            return self

        from hmmlearn.hmm import GaussianHMM
        X = (allv / self.scale)[:, None]
        m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=C.HMM_N_ITER,
                        random_state=C.HMM_SEED, init_params="", params="stmc")
        q = np.percentile(X, [10, 90])
        m.startprob_ = np.array([0.1, 0.8, 0.1])
        m.transmat_ = np.array([[0.96, 0.03, 0.01], [0.02, 0.96, 0.02], [0.01, 0.03, 0.96]])
        m.means_ = np.array([[q[0]], [0.0], [q[1]]])
        m.covars_ = np.array([[1.0], [0.05], [1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, lengths=[len(d) for d in ds])
        order = np.argsort(m.means_.ravel())
        self.map = np.empty(3, int)
        self.map[order] = [-1, 0, 1]
        self.model = m
        mu = m.means_.ravel()[order] * self.scale
        sd = np.sqrt(m.covars_.ravel()[order]) * self.scale
        self.info = dict(method="hmm", converged=bool(m.monitor_.converged),
                         means=mu.tolist(), sds=sd.tolist(),
                         stationary_mean_ok=bool(abs(mu[1]) < 0.5 * min(abs(mu[0]), abs(mu[2]))))
        return self

    def decode(self, v):
        v = np.asarray(v, float)
        if self.method == "threshold":
            lab = _hysteresis(v, C.THR_K_ON * self.robust_sd, C.THR_K_OFF * self.robust_sd)
        else:
            vd = v[::_STEP] / self.scale
            if len(vd) < 3:
                return np.zeros(len(v), int)
            lab_d = self.map[self.model.predict(vd[:, None])]
            lab = lab_d[np.minimum(np.arange(len(v)) // _STEP, len(lab_d) - 1)]
        return merge_short(lab, int(C.MIN_SEG_S * C.FS))


def _hysteresis(v, on, off):
    lab = np.zeros(len(v), int)
    cur = 0
    for i, x in enumerate(v):
        if cur == 0:
            cur = 1 if x > on else (-1 if x < -on else 0)
        elif cur * x < off:
            cur = 1 if x > on else (-1 if x < -on else 0)
        lab[i] = cur
    return lab


def runs(lab):
    """(start, stop, value) of the runs of equal values; stop is exclusive."""
    lab = np.asarray(lab)
    if len(lab) == 0:
        return []
    ch = np.where(np.diff(lab) != 0)[0] + 1
    st = np.r_[0, ch]
    sp = np.r_[ch, len(lab)]
    return list(zip(st, sp, lab[st]))


def merge_short(lab, min_len):
    lab = np.asarray(lab).copy()
    for _ in range(50):
        r = runs(lab)
        short = [(i, a, b) for i, (a, b, _) in enumerate(r) if b - a < min_len]
        if not short or len(r) == 1:
            break
        for i, a, b in short:
            fill = r[i - 1][2] if i > 0 else r[i + 1][2]
            lab[a:b] = fill
    return lab


def segment_table(lab, t, pos, kind, who):
    """One row per segment: state, onset, offset, duration, amplitude."""
    rows = []
    for a, b, v in runs(lab):
        rows.append(dict(signal=kind, who=who, state=int(v),
                         state_name=STATE_NAMES[kind][int(v)],
                         onset=float(t[a]), offset=float(t[b - 1]),
                         duration=float(t[b - 1] - t[a]),
                         amplitude=float(pos[b - 1] - pos[a])))
    return rows


def interaction_labels(s1, s2):
    """Paper definitions on the vertical hand states.
    aligned : hands move in opposite vertical directions (rotate the beam the same way)
    opposed : same vertical direction (oppose each other's rotation)
    single  : exactly one hand moves
    stationary : neither moves"""
    s1, s2 = np.asarray(s1), np.asarray(s2)
    out = np.full(len(s1), "stationary", dtype=object)
    out[(s1 == 0) ^ (s2 == 0)] = "single"
    out[(s1 != 0) & (s2 != 0) & (s1 == -s2)] = "aligned"
    out[(s1 != 0) & (s2 != 0) & (s1 == s2)] = "opposed"
    return out


def interaction_metrics(s1, s2, y1, y2):
    lab = interaction_labels(s1, s2)
    n = len(lab)
    m = {f"ratio_{k}": float(np.mean(lab == k)) for k in INTERACTION}
    # unilateral manipulation ratio: vertical path length while only one hand moves
    dy1, dy2 = np.abs(np.diff(y1, prepend=y1[0])), np.abs(np.diff(y2, prepend=y2[0]))
    only1 = (np.asarray(s1) != 0) & (np.asarray(s2) == 0)
    only2 = (np.asarray(s2) != 0) & (np.asarray(s1) == 0)
    d1, d2 = dy1[only1].sum(), dy2[only2].sum()
    m["unilateral_dist_P1_mm"] = 1e3 * float(d1)
    m["unilateral_dist_P2_mm"] = 1e3 * float(d2)
    m["unilateral_ratio_P1"] = float(d1 / (d1 + d2)) if d1 + d2 > 0 else np.nan
    m["n_interaction_switches"] = int(np.sum(lab[1:] != lab[:-1])) if n > 1 else 0
    return m, lab
