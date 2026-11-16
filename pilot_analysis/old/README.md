# Dyadic beam-balancing pilot analysis

## Files
| File | Role |
|---|---|
| `config.py` | **Edit this only**: dyad folders + protocol, output paths, processing parameters |
| `data_utils.py` | Loading, clock sync, phase labelling, segmentation, filtering |
| `metrics.py` | SPARC, lead-lag, energy leadership, corrective forces, variability |
| `plot_utils.py` | Shared plotting (P1 = solid, P2 = dashed) |
| `run_analysis.py` | Per-dyad analysis -> `Results/<id>/` |
| `overlay_analysis.py` | Across-dyad summary -> `Summary/ProtocolA`, `Summary/ProtocolB`, `Summary/Pooled` |

`all_pilots.py` is superseded by `overlay_analysis.py` and can be deleted.

## Run
```
python3 run_analysis.py               # all dyads
python3 run_analysis.py Pilot2        # one dyad
python3 overlay_analysis.py           # group summaries (independent of the above)
```
Put the scripts next to the pilot folders (or use absolute paths in `config.PILOTS`).
Requires numpy, scipy, pandas, matplotlib.

## Phases
Phases are detected from the chronological order of the `condition` column:
each new run of Baseline trials becomes Baseline 1, Baseline 2. The detected
sequence is checked against the protocol you assign in `config.PILOTS`
(A: Baseline 1, C1, C2 / B: Baseline 1, C1, Baseline 2, C2) and printed in
`qc_report.txt`. Rest periods are not trials.

## Sign conventions
**Every leadership metric is positive when Player 1 (Primary) leads.**
Tilt is positive when P1's end of the beam is higher. dx = x1 - x2.

## Metrics (per trial, in `trial_summary.csv`)
| Your metric | Columns | Definition |
|---|---|---|
| Trajectory | `traj_dev_P1/P2`, `path_len_P*` | RMSE of the time-normalised path from the dyad's own Baseline-1 mean path |
| Hand position difference | `mean_abs_dx`, `peak_abs_dx`, `rms_dx`, `mean_dx` | x1 - x2 during the movement |
| Variability | `var_x1`, `var_x2`, `var_v*`, `var_tilt`, `endpoint_sd_P*`, `cv_movement_time` (phase level) | mean across the 0-100 % grid of the across-trial SD |
| Smoothness | `sparc_P1`, `sparc_P2`, `sparc_object`, `n_submov_P*` | SPARC (Balasubramanian 2015) on speed between onset and offset; closer to 0 = smoother |
| Force (logged) | `mean_abs_F1/F2`, `force_asym_idx`, `mean_abs_haptic` | |
| Force (kinematic) | `rms_kin_force_P*` | m_eff * acceleration (m_eff in config) |
| Corrective forces | `corr_force_impulse_P1/P2`, `corr_force_share_idx`, `corr_disp_share_idx` | Impulse of the tilt-reducing vertical kinematic force (P1 moving down / P2 up when P1's end is high); share > 0 = P1 corrects more |
| Lead-lag | `xcorr_lag_vel`, `xcorr_lag_vy`, `frac_time_P1_leads`, `n_lead_switches` | Cross-correlation of velocities (sub-sample peak), along the path and vertically; time-resolved in 0.5 s windows |
| Beam tilt stability | `max_abs_tilt`, `rms_tilt`, `sd_tilt`, `pct_time_level`, `n_tilt_corrections`, `slope_*` (phase level) | |
| Velocity | `peak_speed_P*`, `mean_speed_P*`, `peak_speed_lag` | |
| Added: energy-flow leadership | `energy_leader_idx_kin` (`_haptic`) | Energy the coupling delivers to each partner; the one being dragged is the follower. Stiffness-independent |
| Added: timing | `onset_lag_kin`, `init_time_diff`, `target_time_diff` | Kinematic and event-based onset/arrival lags |
| Added: conflict | `frac_opposing_motion` | Fraction of moving time the hands move in opposite directions |
