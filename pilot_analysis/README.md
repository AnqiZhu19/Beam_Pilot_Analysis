# Dyadic beam-balancing pilot analysis

## Files
| File | Role |
|---|---|
| `config.py` | **Edit this only**: dyad folders + protocol, robot parameters, processing settings |
| `data_utils.py` | Loading, clock sync, phase labelling, segmentation, filtering, coupling fit |
| `metrics.py` | All metric definitions |
| `plot_utils.py` | Plotting + `FIG_SPECS`: the metric list of each figure, shared by both scripts |
| `run_analysis.py` | Per dyad -> `Results/<id>/` |
| `overlay_analysis.py` | Across dyads -> `Summary/ProtocolA`, `ProtocolB`, `Pooled`, `AllPhases_byProtocol` |
| `diagnose.py` | Low-memory check of the raw files (rows, sampling, logging windows) |

```
python3 diagnose.py            # optional: check the raw files first
python3 run_analysis.py        # all dyads (or: python3 run_analysis.py Pilot2)
python3 overlay_analysis.py    # group figures (does not need run_analysis first)
```

## Outputs
Per dyad (`Results/<id>/`): `trial_summary.csv` (one row per trial, every metric),
`phase_summary.csv` (phase means + variability + learning slopes),
`early_late_summary.csv`, `qc_report.txt`, `figures/`.

Figures (same numbers in `Results/` and `Summary/`; in `Summary/` dyads are
overlaid: each coloured dot = one dyad's mean +/- SD over its trials, black
diamond = mean of dyad means +/- SEM across dyads):

| Fig | Content |
|---|---|
| 00 | QC: example raw trial (per dyad only) |
| 01 | Trajectories, 0-100 % movement-normalised (shape comparison) |
| 01b | Trajectories in **real time** from movement onset; dotted line/grey band = mean +/- SD completion time |
| 02 | Velocity |
| 03 | Hand-position difference |
| 04 | Every key metric across trials (learning curves) |
| 05 | Beam tilt |
| 06 | Smoothness |
| 07 | Forces, viscous-field model: interaction, participant, viscous effort, logged |
| 07b | Corrective forces (Eq. 7) |
| 08 | Leader-follower summary |
| 09 | Trial-to-trial variability |
| 10 | First 5 vs last 5 trials of each phase |
| 11 | Relative phase |
| 12 | Phase means per dyad, connected across phases (overlay only) |
| 13 | Lead-lag cross-correlation function r(tau) |
| 13b | Lead-lag along the movement (windowed cross-correlation heatmap) |

## Conventions
* P1 = Primary, P2 = Secondary. P1 solid lines, P2 dashed.
* x = movement (path) axis, y = vertical. Forces: **+ = towards +axis**.
* **Every leadership metric is positive when P1 leads.**
* Tilt > 0 when P1's end of the beam is higher. dx = x1 - x2.
* The dyad is the unit of analysis in the group figures (mean over trials first).

---

# How to read each metric

### Trajectory
* **Normalised trajectory (fig 01)** - shape of the path independent of speed.
  Two players whose curves overlap move as one unit; a gap means one is ahead.
* **Real-time trajectory (fig 01b)** - the same in seconds. Mean curve drawn only
  while >= 50 % of trials are still moving. The title gives completion time.
* `movement_time` - first leaving home to the trial `end` event (s).
* `traj_dev_P1/P2` - RMS distance of the normalised path from the dyad's own
  Baseline-1 mean path. ~0 in Baseline 1; a rise in C1/C2 = the CoM shift
  distorts the movement; a decline across trials = adaptation; values in
  Baseline 2 above Baseline 1 = after-effect.

### Hand-position difference (fig 03)
* `mean_abs_dx`, `peak_abs_dx`, `rms_dx` - how far apart the hands are along the
  path. Larger = less spatial agreement (with a spring coupling, also more force
  exchanged).
* `mean_dx` (signed) - who is ahead on average (> 0: P1 further along +x).
  If the task moves towards -x, the sign flips; check fig 01.
* `frac_opposing_motion` - fraction of moving time the hands move in opposite
  directions (explicit conflict / tug-of-war).

### Velocity (fig 02)
* `peak_speed_P*`, `mean_speed_P*` - vigour.
* `peak_speed_lag` = t_peak(P2) - t_peak(P1); > 0 P1 peaks first.
* Relative velocity v1 - v2 curve: > 0 where P1 moves faster (pulling ahead).

### Variability (fig 09, phase level)
* `var_x1`, `var_x2`, `var_v*`, `var_tilt`, `var_dx` - mean over the normalised
  grid of the across-trial SD: how repeatable the movement is. Expect high
  early in a new condition, decreasing with learning.
* `endpoint_sd_P*` - spread of the final position; `cv_movement_time` - timing
  consistency (SD / mean).
* Fig 10 / `var_*` early vs late: a drop = a stable strategy has formed.

### Smoothness (fig 06)
* `sparc_P1`, `sparc_P2`, `sparc_object` - spectral arc length of the speed
  profile (Balasubramanian et al. 2015). **Negative; closer to 0 = smoother.**
  A single minimum-jerk reach is about -1.4; corrections push it towards -2, -3.
  `sparc_object` uses the beam-centre velocity (the shared object).
* `n_submov_P*` - number of speed peaks (submovements/corrections; 1 = one
  smooth reach). `sparc_tilt_rate` - smoothness of the balance control.

### Beam tilt (fig 05)
* `max_abs_tilt`, `rms_tilt` - balance error (deg); `mean_tilt` (signed) - a
  systematic lean, e.g. towards the CoM side; `sd_tilt` - fluctuation in a trial.
* `pct_time_level` - % of time within +/- `TILT_TOLERANCE_DEG` (performance).
* `n_tilt_corrections` - tilt reversals > 1 deg (discrete balance corrections).
* Stability over trials: fig 04 / fig 05 and `slope_max_abs_tilt`,
  `slope_rms_tilt` in `phase_summary.csv` (negative slope = improving).

### Forces (fig 07, 07b) - all signed, viscous-field model
No force sensors and unknown robot dynamics, so the forces are inferred from
what the robot renders. Each handle feels a viscous field (-B_visc v) and the
coupling spring to the partner (K_c (x_j - x_i) + B_c (v_j - v_i)). If the
acceleration term is negligible, the hand is in equilibrium and the participant's
force is equal and opposite:

    F_human_i = B_visc * v_i  +  K_c (x_i - x_j) + B_c (v_i - v_j)
                (effort against      (interaction force: what i
                 the viscous field)    transmits to the partner)

* Gains: set `VISCOUS_B` and `COUPLING` in `config.py` to the values in your
  controller. If left at None they are estimated per dyad by regression
  (B_visc from ToolForce, K_c/B_c from HapticForce or F1); the fits and R^2 are
  in `qc_report.txt`, and fits with R^2 < 0.5 are not used.
* Validity: the mass term is negligible only if the robot's apparent mass is much
  smaller than B v / a; `qc_report.txt` prints that limit (kg).
* `mean_Fint_x/y` - signed interaction force from P1 to P2 (P2 receives it, and
  sends back -F_int). `mean_abs_Fint_*` = how much they push/pull on each other.
* `Fint_lead_x` - interaction force along the direction of motion: > 0 P1 pulls
  or pushes P2 along (**P1 leads**), < 0 P1 is held back / dragged.
  `frac_P1_pulls_x` - fraction of the interacting time P1 is the one pulling.
* `mean_Fh_*_P1/P2` - each participant's total force; `mean_Fvisc_*` - the part
  spent against the viscous field (moving the load).
* `work_x_P*` = integral of F_human v: > 0 drives the motion, < 0 brakes it.
* `dominance_P1_x` - share of the object-moving (effective) force produced by P1
  (Groten et al. 2009), > 0.5 P1 dominant.
* `mean_F1/F2`, `load_share_idx` - the logged commanded forces, for reference.

**Corrective forces (fig 07b), Eq. 7**:

    F_nc = (1/N) sum_t | F(t) - SMA_k(F)(t) |

the mean deviation of the handle force from its own simple moving average:
fast corrective adjustments around the slow force trend. `CORRECTIVE_SMA_WINDOW_S`
sets k (use the paper's value). The printed equation sums j = 0..k but divides
by k; a true average over the window is used.
* `Fnc_y_P1`, `Fnc_y_P2` - per handle, from each participant's force F_human
  (vertical, as in the paper; `Fnc_x_*` on the path axis too). If no vertical
  gains are available, the logged vertical forces F1/F2 are used (flagged in the
  qc report). `Fnc_share_*` > 0: P1 makes more corrections.
* `Fnc_int_*` - Eq. 7 on the interaction force. One value per dyad: the
  interaction force is equal and opposite for the two players, so with the
  absolute value it cannot differ between them.

### Lead-lag and temporal coordination (fig 08, fig 11)
* **Lead-lag cross-correlation (fig 13)**: r(tau) = correlation between P1's
  signal and P2's signal shifted by tau, plotted for tau = -0.4 .. +0.4 s.
  The tau of the peak is the lead/lag (> 0: P2 repeats P1 tau seconds later,
  P1 leads), the peak height how strongly they are coupled (1 = same shape).
  `xcorr_lag_vel` (s), `xcorr_lag_vel_pct` (% of movement time, comparable
  across speeds), `xcorr_r_vel`. Computed on velocities, not positions: two
  ramps correlate at every lag, which flattens the peak.
* **Lead-lag along the movement (fig 13b)**: the same r(tau) in overlapping
  0.5-s windows centred at 10 points of the movement (heatmap, black = peak
  lag per window). Shows whether the leader is constant or changes, e.g.
  P1 initiates and P2 brakes.
* `xcorr_lag_vy` - same on vertical velocities (who moves the beam end first).
* `frac_time_P1_leads` - same lag in sliding 0.5-s windows; fraction of windows
  where P1 leads (0.5 = shared/alternating; near 0 or 1 = a fixed leader).
  `n_lead_switches` - how often the leader changes within a trial (role
  negotiation vs a settled role).
* `energy_leader_idx_kin` - energy flowing through the coupling: the partner
  being dragged receives energy. > 0: P2 is dragged, P1 leads. K-independent.
* `onset_lag_kin`, `init_time_diff` (leave-home event), `target_time_diff`,
  `peak_speed_lag` - discrete timing: who starts / peaks / arrives first.
* **Relative phase** (fig 11):
  * `crp_mean_deg` - continuous relative phase. Each player's movement phase is
    read from their normalised position-velocity plane (0 deg start, 90 deg peak
    velocity, 180 deg end); CRP = phase(P1) - phase(P2). > 0: P1 is further
    through the movement (leads). Unlike the lag (seconds), phase is relative to
    the movement cycle, so it is comparable across trials of different speed.
  * `crp_sd_deg` - circular SD of the CRP within the trial: coordination
    stability (lower = more tightly coupled timing).
  * `relphase_vel_deg`, `plv_vel` and `relphase_vy_deg`, `plv_vy` - Hilbert
    relative phase (> 0: P1 leads) and phase-locking value (0 = no consistent
    relation, 1 = perfectly locked). The vertical one suits the oscillatory
    balance corrections; the velocity one the reach.
* How to combine them: a convincing leader shows the same sign across
  timing (lag, CRP), energy and force dominance. Disagreement is informative,
  e.g. one partner leads in time while the other generates the force.

### What to look for in this design
* **CoM shift (C1, C2)**: does the near player carry more load (`load_share_idx`)
  and become the leader (positive lead metrics in C1, negative in C2)?
* **Adaptation**: within-phase slopes of tilt, dx, movement time, SPARC, and
  variability early vs late.
* **After-effect (protocol B)**: Baseline 2 vs Baseline 1 in trajectory deviation,
  tilt and lead metrics.
* **Role transfer**: C2 in protocol B (after washout) vs protocol A (directly
  after C1); n = 1 for A, so descriptive only.

With 4 dyads in protocol B treat everything as descriptive.

## Before running on real data
* Set `VISCOUS_B`, `COUPLING` (and `CORRECTIVE_SMA_WINDOW_S`) in `config.py`
  to your controller's values, or check the fitted ones in `qc_report.txt`.
* Read each `qc_report.txt`: sync matches, coupling-fit R^2 per axis, which force
  source was used, whether the two robots' position logs agree, median tilt.
* If the Secondary robot's frame is mirrored, set `SECONDARY_X_SOURCE` and
  `SECONDARY_FORCE_SIGN`.
