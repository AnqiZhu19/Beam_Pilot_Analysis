"""
Configuration for the new (Nov 2026) dyadic beam pilot analysis.

Edit this file (not the analysis code) to change the data folder, block
labels or processing parameters.

Conventions
-----------
P1 = Primary, P2 = Secondary.  x = reach axis (home ~0.20 m -> target ~0.45 m),
y = vertical.  dx = x1 - x2 (> 0: P1 further along the reach).
theta = asin((y1 - y2) / BEAM_LENGTH_M)  [deg], > 0 when P1's end is higher.
Every lead/lag metric is > 0 when P1 leads (moves / arrives first).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
DATASETS = [
    dict(id="eeg_pilot", folder=os.path.join(ROOT, "eeg_pilot")),
]
OUT_ROOT = os.path.join(HERE, "Results")
OLD_CODE_DIR = os.path.join(ROOT, "pilot_analysis")   # reused helpers

# block num -> short label. Blocks not listed (Practice, PracticeHaptic) are skipped.
BLOCKS = {3: "B1", 4: "C1a", 5: "C1b", 6: "B2", 7: "C2"}
BLOCK_ORDER = ["B1", "C1a", "C1b", "B2", "C2"]
BLOCK_LONG = {
    "B1": "Baseline 1",
    "C1a": "C1 block A",
    "C1b": "C1 block B (after 60 s rest)",
    "B2": "Baseline 2",
    "C2": "C2",
}
# categorical slots 3-7 of the validated default palette (slots 1-2 = players)
BLOCK_COLORS = {"B1": "#1baf7a", "C1a": "#eda100", "C1b": "#e87ba4",
                "B2": "#008300", "C2": "#6250d6"}
P1_COLOR, P2_COLOR = "#2a78d6", "#eb6834"      # P1 blue, P2 orange (paper: left/right)
OBJ_COLOR = "#3d3d3a"
SUCCESS_ONLY = True

# ----------------------------------------------------------------------
# SIGNAL PROCESSING  (paper: 10th-order Butterworth, 20 Hz, dual pass)
# ----------------------------------------------------------------------
FS = 500.0
LOWPASS_HZ = 20.0
FILTER_ORDER = 10
MIN_COVERAGE = 0.80          # trials whose log covers < 80 % of [begin, end] are dropped
BEAM_LENGTH_M = 0.60         # virtual beam length between the hands (for theta)
N_GRID = 101                 # 0-100 % movement-normalised grid
RT_FS = 50.0                 # real-time curves: resolution (Hz)
RT_MAX_S = 4.5               # real-time curves: length after movement onset (s)

# reach onset / offset (threshold method, on |vx|)
ONSET_FRAC = 0.05            # fraction of the trial's peak speed
ONSET_MIN = 0.01             # m/s floor
ONSET_SUSTAIN_S = 0.05

# ----------------------------------------------------------------------
# HMM SEGMENTATION (paper Methods: 3-state HMM on velocity)
# ----------------------------------------------------------------------
SEG_METHOD = "hmm"           # "hmm" or "threshold" (hysteresis fallback)
HMM_FS = 100.0               # fit/decode rate (Hz); labels are upsampled to FS
HMM_N_ITER = 200
HMM_SEED = 0
MIN_SEG_S = 0.04             # runs shorter than this are merged into a neighbour
# threshold fallback: |v| > k * robust SD of the signal (with hysteresis)
THR_K_ON, THR_K_OFF = 1.0, 0.5
MIN_SEG_AMP_X = 0.01         # m, X segments smaller than this are not counted as "reach" segments

# corrective movement, paper Eq. 11 (moving average over the past k)
CORRECTIVE_WIN_S = 0.20

# ----------------------------------------------------------------------
# LEAD-LAG
# ----------------------------------------------------------------------
XCORR_MAX_LAG_S = 0.20
XCORR_PHASE_MIN_S = 0.15     # phases shorter than this get no xcorr lag
XCORR_MIN_R = 0.30           # xcorr lags with a weaker peak correlation are set to NaN
TLCC_N_WINDOWS = 10

# ----------------------------------------------------------------------
# EARLY vs LATE
# ----------------------------------------------------------------------
EARLY_N = 10                 # first/last N successful trials of a block (min(N, n//2))
POST_REST_N = 10             # C1b: first N trials after the rest

# example trials for the Fig.8-style plots: "auto" = first & last successful
# trial of every block, or a list of (block, trial) tuples
EXAMPLE_TRIALS = "auto"
