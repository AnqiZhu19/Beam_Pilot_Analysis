"""
Central configuration for the dyadic beam-balancing pilot analysis.

Edit THIS file (not the analysis scripts) when you add a dyad, change a path
or need to adjust a processing parameter.
"""

# ----------------------------------------------------------------------
# DYADS
# ----------------------------------------------------------------------
# id       : short name used for the output folder  (Results/<id>/)
# folder   : raw-data folder containing trial.csv, primarySpringCouplingLog.csv,
#            secondarySpringCouplingLog.csv
# protocol : "A" or "B" (see PROTOCOLS below)
#
# NOTE: the pilot4 / pilot5 folder names end in "_" -> they look truncated.
#       Fix them to the real folder names, otherwise those dyads are skipped.
PILOTS = [
    dict(id="Pilot1", folder="./pilot1_20260810_121516", protocol="A"),
    dict(id="Pilot2", folder="./pilot2_20260810_153312", protocol="B"),
    dict(id="Pilot3", folder="./pilot3_20260810_162302", protocol="B"),
    dict(id="Pilot4", folder="./pilot4_20260812_",       protocol="B"),
    dict(id="Pilot5", folder="./pilot5_20260812_",       protocol="B"),
]

# Expected phase sequence of each protocol (rest periods are not trials and
# are therefore not listed). Phases are detected automatically from the
# chronological sequence of the 'condition' column in trial.csv: every new
# run of "Baseline" trials becomes "Baseline 1", "Baseline 2", ...
PROTOCOLS = {
    "A": dict(
        name="Protocol A: Baseline - C1 - rest - C2",
        phases=["Baseline 1", "C1", "C2"],
    ),
    "B": dict(
        name="Protocol B: Baseline - C1 - rest - Baseline - C2",
        phases=["Baseline 1", "C1", "Baseline 2", "C2"],
    ),
}

# Phases that are identical (same history) in both protocols and can be
# pooled across protocols. C2 is NOT poolable: in protocol B it follows a
# washout baseline, in protocol A it directly follows C1 (+ rest).
POOLABLE_PHASES = ["Baseline 1", "C1"]

PHASE_LABELS = {
    "Baseline 1": "Baseline 1 (CoM centre)",
    "C1": "C1 (CoM at Player 1)",
    "Baseline 2": "Baseline 2 (washout)",
    "C2": "C2 (CoM at Player 2)",
}
PHASE_COLORS = {
    "Baseline 1": "#4C72B0",
    "C1": "#DD8452",
    "Baseline 2": "#8DA9D4",
    "C2": "#55A868",
}

# Condition names in trial.csv that are NOT trials (ignored if present)
NON_TRIAL_CONDITIONS = {"Rest", "rest", "REST", "Break", "break"}

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------
RESULTS_ROOT = "./Results"   # Results/<id>/figures, Results/<id>/*.csv
SUMMARY_ROOT = "./Summary"   # Summary/ProtocolA, Summary/ProtocolB, Summary/Pooled

# ----------------------------------------------------------------------
# SIGNAL PROCESSING
# ----------------------------------------------------------------------
FS = 500.0               # Hz, uniform resampling rate (logs are ~480-500 Hz)
LOWPASS_HZ = 10.0        # zero-phase 4th-order Butterworth before differentiation
MAX_GAP_S = 0.05         # a data gap longer than this inside a trial is flagged
N_GRID = 101             # points of the 0-100 % movement-normalised grid

# Movement onset/offset from speed (per player)
ONSET_SPEED_FRAC = 0.05  # fraction of the trial's peak speed
ONSET_SPEED_MIN = 0.01   # m/s, absolute floor for the threshold

# Which signal is used for the Secondary's X / the Secondary's height:
#   "secondary_file": Secondary MyPosX/MyPosY from secondarySpringCouplingLog
#   "primary_other" : OtherPosX/OtherPosY from primarySpringCouplingLog
# The QC report prints how well the two agree; if the robots use different
# frames (e.g. mirrored X), use "primary_other" so both hands share a frame.
SECONDARY_X_SOURCE = "secondary_file"
SECONDARY_Y_SOURCE = "primary_other"   # as in the original beam-angle code

# ----------------------------------------------------------------------
# TASK / PHYSICS
# ----------------------------------------------------------------------
BEAM_LENGTH_M = 0.60     # distance between the two hands on the (virtual) beam
TILT_TOLERANCE_DEG = 5.0 # "beam level" band used for time-in-tolerance
M_EFF_KG = 1.0           # effective mass per hand for kinematic force (F = m*a);
                         # with 1 kg the kinematic forces are in N per kg.

# ----------------------------------------------------------------------
# LEADER-FOLLOWER
# ----------------------------------------------------------------------
XCORR_MAX_LAG_S = 0.40   # search range for the cross-correlation lead-lag
WIN_LEADLAG_S = 0.50     # sliding window for time-resolved lead-lag
WIN_STEP_S = 0.10
WIN_MIN_CORR = 0.50      # windows with lower peak correlation are ignored

# ----------------------------------------------------------------------
# LEARNING / EARLY vs LATE
# ----------------------------------------------------------------------
EARLY_N = 5
LATE_N = 5

# Colours
PILOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
P1_LS, P2_LS = "-", "--"   # Player 1 (Primary) solid, Player 2 (Secondary) dashed
