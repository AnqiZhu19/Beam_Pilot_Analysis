# analysis_v2: analysis of the Nov 2026 pilot (`eeg_pilot/`)

This is a new pipeline for the new `trial.csv` format: blocks, `begin/running/end` events, success/fail results, and no left-home events. It follows Liu et al. 2025 (*Partner haptic feedback*, Fig. 8 and Methods). The old `pilot_analysis/` code is left unchanged, and some of its helpers are imported: `detect_time_offset`, `sparc`, `n_submovements`, `xcorr_*` and `tlcc_matrix`.

```
python3 -m pip install hmmlearn        # once; installs into the python3 that runs the script
python3 analysis_v2/run_v2.py
open analysis_v2/Results/eeg_pilot/figures     # view the figures in Finder
```
Results are saved in the folder `analysis_v2/Results/eeg_pilot/` (CSVs) and its `figures/` subfolder.
To change the data folder, block labels, filter or thresholds, edit **`config_v2.py`**.

## Data handling
* Blocks: 3 = **B1** (Baseline), 4 = **C1a**, 5 = **C1b** (C1 after the 60 s rest), 6 = **B2**, 7 = **C2** (only 20 trials). Practice blocks are skipped.
* Only **successful** trials are analysed. Failure rates per block are in `failure_rate.csv`.
* The robot logs only record during trials, on a clock offset from trial.csv. The offset is detected automatically (5.638 s here, all 295 trials matched). See `qc_report.txt`.
* Signals are resampled to 500 Hz and low-pass filtered with a zero-phase 10th-order Butterworth at 20 Hz (as in the paper).
* x is the reach axis (home ≈ 0.20 m → target ≈ 0.45 m), y is vertical, and θ = asin((y1 − y2)/0.6 m), with θ > 0 when P1's end is higher.
* **Sign convention: every lead/lag/difference metric is > 0 when P1 (Primary) is first or ahead.**

## Segmentation (paper Methods: "Movement segmentation")
Three 3-state Gaussian HMMs are fitted on velocity, pooled over all trials and both players, then decoded with Viterbi. Segments shorter than 40 ms are merged into a neighbour.

| model | signal | states |
|---|---|---|
| vert | ẏ1, ẏ2 | down / stationary / up (paper Fig. 8a) |
| beam | θ̇ | cw / stationary / ccw (Fig. 8c) |
| reach | ẋ1, ẋ2 | backward / stationary / forward (*our* addition, used for timing) |

**Interaction dynamics** are computed from the two hands' vertical states, as in Fig. 8b:
* **aligned**: the hands move in opposite vertical directions, so both rotate the beam the same way.
* **opposed**: the hands move in the same direction.
* **single**: only one hand moves.
* **stationary**: neither hand moves.

Each ratio is the time spent in that category divided by the trial time. `F9_hmm_states.png` shows the fitted states, and the means are in `qc_report.txt`.

## Outputs
| file | content |
|---|---|
| `trial_summary.csv` | one row per trial, every metric |
| `block_summary.csv` / `block_means_wide.csv` | per block: mean, SD, median, slope over trials |
| `early_late.csv` | first vs last 10 trials of each block (5 for C2) |
| `variability.csv` | across-trial SD of the movement-normalised curves (all / early / late), endpoint SD, CV of time |
| `C1_post_rest.csv` | C1a first/last 10 vs C1b first 10 (post-rest) / last 10 |
| `segments.csv` | every HMM segment (signal, player, state, onset, offset, amplitude) |
| `figures/F0_example_trials/` | **Fig. 8 replica**: (a) vertical segmentation, (b) interaction dynamics, (c) beam rotation, (d) reach with movement phases, (e) reach segmentation |
| F1 / F1b | trajectories x, dx, y, θ, ẋ per block (normalised / real time) |
| F2 / F2b | position difference, corrective movements; variability, failure rate |
| F3 | smoothness (SPARC, submovements) |
| F4 | beam tilt, θ–velocity correlations, rotation segments, unilateral ratio |
| F5 | interaction-dynamics ratios, coordination polygon, composition over time |
| F6 | initiation / completion / delays, C1 around the rest |
| F7 / F7_tlcc | lead-lag per movement phase; windowed cross-correlation heatmap |
| F8 | learning curves |

In the per-block panels, dots are trials, ● is the mean ± SD, ◁ is the mean of the first 10 trials and ▷ the mean of the last 10.

## Metrics
**Trajectory**
* `mean_dx_mm`, `mean_abs_dx_mm`, `rms_dx_mm`, `peak_abs_dx_mm` are the position difference along the reach, measured over the movement window. `end_dx_mm` is the final offset.
* `corr_{y,x}_P*_mm` is corrective movement (paper Eq. 11): the mean |z(t) − average of z over the past 200 ms|. `CR_{y,x}_P1` = C1/(C1 + C2), so values > 0.5 mean P1 makes more corrections.
* `n_backward_P*` counts backward X segments of at least 1 cm (overshoot corrections).
* `sparc_P1/P2/obj` measure smoothness and are closer to 0 when smoother. `n_submov_*` counts speed peaks.
* Variability is in `variability.csv` / F2b.

**Beam tilt**
* `max_abs_theta`, `rms_theta`, `mean_theta`, `end_theta`.
* `r_theta_objv` = r(θ, beam-centre X velocity); `r_theta_dx` = r(θ, x1 − x2); `r_absthetadot_speed` = r(|θ̇|, speed).
* `lag_objv_theta` is the cross-correlation lag between velocity and θ (> 0 means velocity changes before tilt). It is NaN when the peak falls at the ±0.5 s edge.
* `r_theta_thetadot` is ≈ 0 by construction for any oscillating signal. It is reported only because it was requested; read the other correlations instead.
* `ratio_*`, `n_rot_segments`, `mean_rot_dur`, `frac_time_rotating` come from the segmentation.
* `unilateral_ratio_P1` is P1's share of the vertical path length during *single* periods (paper's unilateral manipulation ratio).

**Temporal ("delay in our way", on the X reach)**
* `init_diff` = reach onset P2 − P1. Both hands use one shared speed threshold (5 % of the larger peak).
* `complete_diff` = last-movement offset P2 − P1.
* `target_entry_diff` = final target entry P2 − P1, taken from trial.csv.
* `P1_initiates`, `P1_enters_target`: 1 if P1 was first.
* `seg_delay_{all,fwd,bwd}` / `seg_absdelay_*` are paired-segment delays. Each X-reach segment of P1 is paired with an overlapping P2 segment in the same direction, and the delay is onset P2 − onset P1. This is the reach-axis analogue of the paper's delay. `fwd` = forward pushes, `bwd` = back-corrections. `seg_frac_P1_first` is the fraction of pairs in which P1 started first.
* `hmm_init_diff` / `hmm_complete_diff` use each player's largest forward HMM segment.
* **Movement phases**:
  * `acc`: onset → peak beam-centre velocity
  * `dec`: peak velocity → first player's final target entry
  * `final`: first entry → trial end
  * For each phase: `*_mean_dx_mm`, `*_frac_P1_ahead`, `*_aligned`, `*_corr_y_P*_mm` and `*_lag_vx`.
* `xcorr_lag_vx` and `*_lag_vx` are cross-correlation lags of ẋ1 and ẋ2. They are kept only when the peak r ≥ 0.3 and the peak is not at the ±200 ms edge.

## Caveats
* **One dyad**, so everything is descriptive. C2 has 19 successful trials.
* **Cross-correlation lags are mostly unusable here.** P1 makes a fast reach with overshoot and back-corrections while P2 moves slowly and steadily, so ẋ1 and ẋ2 correlate weakly (r ≈ 0). Only 0–23 trials per block pass the gates. Use the event- and segment-based timing measures (`init_diff`, `target_entry_diff`, `seg_delay_*`) instead.
* Trials end as soon as both players are in the target, so there is no hold period. The "final" phase is the time from one player's arrival to the other's.
* `BEAM_LENGTH_M = 0.6` only scales θ. Set it to the real virtual beam length if it differs.
* The vertical HMM is shared by both players, so state definitions are identical. P2's vertical movements are small, so P2 is more often "stationary". This is a real asymmetry, not a fitting artefact; check F9 to confirm.
