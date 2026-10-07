# Keystroke Sampling Latency vs. Detection Accuracy

Reproducible pipeline for: *Optimising Real-Time Behavioural Biometrics: A
Sensitivity Analysis of Sampling Latency vs. Detection Accuracy in Banking
Fraud* (COS700).

Evaluates how downsampling keystroke timing data (simulating coarser
hardware polling, e.g. 1ms -> 20ms -> 30ms) affects authentication accuracy
(EER/FAR/FRR) across three anomaly-detection models, and projects the
corresponding AWS infrastructure cost savings.

## Setup

```bash
pip install -r requirements.txt
```

Requires: `pandas`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`,
`tensorflow`.

## Data

The real CMU Keystroke Dynamics Benchmark Dataset
(`DSL-StrongPasswordData.csv`, Killourhy & Maxion, 2009) lives in
`data/raw/`. The downsampled group CSVs in `data/processed/` are not
tracked in git; step 2 below regenerates them deterministically.

## Training and evaluation protocol (important - read before running)

Each subject's model is trained **once**, using only that subject's
`control_1ms` (1ms, full-resolution) data (the first 200 of their 400
repetitions). That same trained model is then evaluated against genuine and
impostor test samples **from each sampling group separately** (the same
held-out repetitions, but downsampled to each group's resolution: 1ms,
20ms, and 30ms).

In other words: training always happens on high-resolution data, and the
question being tested is "does a model trained on precise timing data still
recognise the user correctly when it's later fed coarser, downsampled
timing data?" This mirrors a realistic deployment scenario: a bank doesn't
retrain a customer's behavioural profile from scratch every time it changes
its live-monitoring polling rate; the profile stays as it was, and only the
resolution of the incoming live data changes.

The Hybrid model fuses Isolation Forest and LSTM Autoencoder scores with
**alpha=0.10** by default (10% Isolation Forest weight, 90% LSTM weight),
matching Shaheen & Alomari's (2026) cited fusion formula. Pass `--alpha 0.5`
to any hybrid script to compare against an equal-weighted fusion instead.

## Run order

```bash
# 1. Validate the real dataset
python3 src/load_and_check.py data/raw/DSL-StrongPasswordData.csv

# 2. Build the three core sampling groups (1ms / 20ms / 30ms) + validation plot
python3 src/downsample.py

# 3. Sanity-check the EER utility against known synthetic cases
python3 src/eer.py

# 4. Train + evaluate each model (train once per subject, test against all 3 groups)
python3 src/train_isolation_forest.py
python3 src/train_lstm_autoencoder.py
python3 src/train_hybrid.py                    # alpha=0.10 by default

# 5. Build the unified 5-metric comparison table (EER/FAR/FRR/latency/data volume)
python3 src/build_comparison_table.py          # alpha=0.10 by default

# 6. Multi-seed run: every resolution (1-500ms) x 10 phase-jitter seeds, per subject
python3 src/run_multiseed_experiment.py

# 7. Test the three pre-specified hypotheses (paired significance tests; uses the 5-training-seed
#    results in results/training_seeds_eer_by_subject.csv when present, else step 6's output)
python3 src/hypothesis_verdicts.py

# 8. Report figures (from step 7's outputs) -> results/figures/
python3 src/make_figures.py

# 9. [Exploratory, beyond H1-H3, single jitter seed] Practical degradation breaking point
python3 src/breaking_point_analysis.py         # tests 1-500ms, ~18 resolutions

# 10. Infrastructure cost projection (grounded in real measured event duration/payload)
python3 src/cost_projection.py

# 11. [Sensitivity analysis] Does any hybrid fusion weight beat Isolation Forest alone?
python3 src/hybrid_alpha_sweep.py              # alpha 0.0-1.0, 3 core resolutions x 10 seeds
```

The training-seed robustness run (`src/run_training_seeds.py`, ~3 hours: retrains every
subject under five training seeds) can be run on Kaggle instead of locally; see
[kaggle/README.md](kaggle/README.md).

Fast checks of the building blocks (no model training, a few seconds):

```bash
python3 -m unittest discover tests -v
```

Steps 4-6, 9 and 11 each retrain models per subject; expect ~35-40 minutes per
script on CPU (step 6 longer, since it scores 171 test conditions per
subject). Training happens once per subject regardless of how many
resolutions or jitter seeds are tested — only the scoring step scales with
them. Use `--n-subjects N` on any script for a quick smoke test on a subset
of subjects before running the full 51.

Steps 4, 5 and 9 use a single phase-jitter seed per resolution, so their
20ms/30ms figures differ from each other and from step 6 by ~0.2-0.3 EER
points. Step 6 averages that noise out and is the source for the
hypothesis verdicts.

## File overview

| File | Purpose |
|---|---|
| `src/load_and_check.py` | Validates the real CMU CSV (shape, missing values, duplicates, timing ranges) |
| `src/downsample.py` | Sliding-window downsampling protocol (quantisation + phase jitter); builds the 3 core group CSVs |
| `src/eer.py` | EER/FAR/FRR computation utility, self-tested against known synthetic cases |
| `src/train_isolation_forest.py` | Isolation Forest: trains once per subject on control data, tests against all 3 groups |
| `src/train_lstm_autoencoder.py` | LSTM Autoencoder (11-timestep sequence framing): same train-once-test-against-all-groups protocol |
| `src/train_hybrid.py` | Score-level fusion of the above two (alpha=0.10 default) |
| `src/build_comparison_table.py` | Single-pass run producing the full EER/FAR/FRR/latency/data-volume-reduction table across all 3 models x 3 groups |
| `src/subject_models.py` | Shared protocol + model code used by every experiment script: train/test split, Isolation Forest, LSTM Autoencoder, hybrid fusion |
| `src/run_multiseed_experiment.py` | Repeats every resolution over 10 phase-jitter seeds; writes per-subject EERs for the significance tests |
| `src/hypothesis_verdicts.py` | Tests H1-H3 as pre-specified, with paired bootstrap CIs and Wilcoxon signed-rank tests; also writes the degradation curve with CIs |
| `src/make_figures.py` | Report figures: EER curve, degradation vs the 5pt threshold, and the H2 sensitivity difference, all with 95% CIs |
| `src/breaking_point_analysis.py` | Exploratory: extends testing to 50-500ms to find where accuracy genuinely breaks down |
| `src/cost_projection.py` | AWS API Gateway (WebSocket) cost projection for a sample-streaming and an event-driven client; event duration and payload size measured directly from real data, not assumed |
| `src/run_training_seeds.py` | Repeats the multi-seed experiment under several model training seeds; resumable |
| `kaggle/` | Self-contained notebook and instructions for running the training-seed experiment on Kaggle |
| `src/hybrid_alpha_sweep.py` | Sensitivity analysis: sweeps the hybrid fusion weight from LSTM-only to Isolation-Forest-only |
| `tests/test_pipeline.py` | Unit tests for EER, downsampling, the train/test split, fusion, the statistics helpers and the cost model |

## Known/expected results (from a full 51-subject run)

Useful as a sanity check that your own run reproduces correctly:

Mean EER over 51 subjects, 5 training seeds and 10 phase-jitter seeds
(`results/training_seeds_eer_by_subject.csv`, produced on Kaggle; see `kaggle/`).
95% CIs are paired bootstrap intervals over subjects (step 7):

- Isolation Forest EER: 8.82% (1ms) -> 9.49% (20ms) -> 10.26% (30ms)
- LSTM Autoencoder EER: 11.72% -> 12.17% -> 12.82%
- Hybrid EER: 11.17% -> 11.62% -> 12.28%
- H1: PARTIALLY SUPPORTED. Degradation stays within the 5pt threshold at 20ms as predicted, but also at 30ms, where H1 predicted it would exceed it (max degradation +1.44pts, 95% CI [+1.21, +1.71], Isolation Forest at 30ms)
- H2: REJECTED. Isolation Forest degrades significantly *more* than LSTM at both 20ms (difference 0.21pts, p=0.035) and 30ms (0.34pts, p=0.035), opposite of hypothesised. Same direction under all 5 training seeds, individually significant in 2 of them. Exploratory: the direction reverses from 55ms onwards, where LSTM degrades significantly more
- H3: CONFIRMED (95% data volume reduction at 20ms holds by construction; the empirical part is that 20ms stays within the 5pt threshold)
- Practical breaking point (5pt threshold, exploratory): mean degradation crosses at 65ms for Isolation Forest and 60ms for LSTM Autoencoder and Hybrid
- Hybrid fusion weight (step 11, seed-42 models): mean EER improves steadily as weight moves toward Isolation Forest (1ms: 12.05% LSTM-only, 11.38% at the adopted 0.10, 9.18% IF-only); no weighting beats Isolation Forest alone
- A single jitter seed varies around these means with a standard deviation of ~0.1-0.2 EER points; a single training seed by 0.45 (Isolation Forest), 0.20 (LSTM Autoencoder) and 0.13 (Hybrid)
- Cost projection (1M authentication events/month), sample-streaming client (one message per poll tick): $2,600/month at 1ms -> $129/month at 20ms (95% savings) -> $86/month at 30ms (96.7% savings)
- Cost projection, event-driven client (one message per key press/release, 22 per password): $22/month at every polling interval, so no saving from a coarser interval

## Notes / gotchas

- Negative UD (up-down latency) values in the sanity check are expected
  (natural keystroke rollover), not a bug.
- `data/processed/*.csv` contain only the 3 core groups (1ms/20ms/30ms).
  The extended breaking-point resolutions (50ms, 55ms, ... 500ms) are
  generated in-memory by `breaking_point_analysis.py` on each run, not
  saved as files.
- All EER/FAR/FRR values in `results/*.csv` are near-identical by
  construction — FAR and FRR are reported at the EER operating threshold,
  where they are equal by definition.
- If you change the hybrid alpha weight, rerun `train_hybrid.py` and
  `build_comparison_table.py` together with the same `--alpha` value so
  results stay consistent with each other.
## License

The code in this repository is released under the MIT License (see `LICENSE`).

The licence covers the code only. `data/raw/DSL-StrongPasswordData.csv` is the
CMU Keystroke Dynamics Benchmark Dataset, created by Kevin Killourhy and Roy
Maxion (Carnegie Mellon University) and included here unchanged for
reproducibility; it remains the work of its authors. If you use it, cite:
Killourhy, K. S. and Maxion, R. A. (2009). Comparing anomaly-detection
algorithms for keystroke dynamics. IEEE/IFIP International Conference on
Dependable Systems & Networks, 125-134. Original source:
https://www.cs.cmu.edu/~keystroke/
