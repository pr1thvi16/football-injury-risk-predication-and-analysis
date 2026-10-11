# Methodology

> Research prototype. **All data are synthetic.** Results are not clinical evidence and are not the IEEE paper's reported results.

## 1. Task
Binary classification: `injury_next_7_days = 1` if a simulated time-loss injury occurs in the 7 days starting on the observation date. One row = one player-week.

## 2. What the paper specifies vs. what this project chose
| Specified (project brief based on the paper) | Implementation choice (assumption) |
|---|---|
| 12,000 synthetic player-week records, ~8.98 % positive | 250 players x 48 weeks; exactly round(0.0898 x 12000) = **1,078** positives |
| 28-day workload, 7-day wellness, static/medical modalities | Channel definitions, simulator, label mechanism and weights |
| Split 8,160 / 1,440 / 2,400 | Player-grouped split (reproduces these counts exactly), stratified-group candidate search |
| Three encoders, concatenation fusion, FC head, sigmoid | GRU + attention, hidden sizes, dropout 0.3, AdamW 1e-3, batch 128, patience 10 |
| LR, RF, GB/XGBoost, MLP, unimodal baselines | Aggregation features for classical models; unimodal models use XGBoost |
| SHAP explainability | Integrated Gradients and permutation importance as complements |

## 3. Synthetic data generation
Per player: latent fitness, **hidden** injury proneness, stress propensity, position, static attributes. Daily workload (weekly microcycle with rest day, build-up, taper, match day; AR(1) periodisation multiplier; random spikes) and wellness (fatigue/soreness accumulate with lag from load; sleep, mood, recovery degrade) are simulated for 28 + 336 days. Observation `w` uses only the 28/7 days **before** its date.

Label: `z = s * f(observed features) + 0.65 * proneness + e`, `e ~ N(0, 0.8)`, then `z + Logistic(0,1)` noise; the top `round(prevalence x N)` records become positives. This equals Bernoulli sampling with `sigmoid(z - c)` conditioned on a fixed total count. `f` combines a U-shaped ACWR term, strain, monotony, previous injuries (count and recency), strength asymmetry, age, 7-day fatigue/soreness/sleep/recovery/stress, a load-spike x poor-recovery interaction and a position offset. The hidden proneness, noise and probabilities are **never saved**. No single feature reproduces the label (tested: best single-feature AUC < 0.70). Associations learned from these labels are properties of the simulator, not evidence of causation.

## 4. Derived workload features (daily `session_training_load = session_rpe x duration_min`)
- acute = mean load over last 7 days; chronic = mean over last 28 days
- ACWR = acute / max(chronic, 1) (a candidate feature only)
- monotony = mean_7d / std_7d (ddof 1), capped at 10; strain = sum_7d x monotony
- variability = std_28d / max(chronic, 1); recent change = (acute - mean of days 15-21 back) / max(that mean, 1)
- average_rest_days = rest days (zero duration) per 7 days over the 28-day window

## 5. Preprocessing and leakage control
Median imputation, standardisation, one-hot position and per-channel sequence scaling are **fit on the training split only**; the preprocessor never reads the target (tested). The default **grouped split** puts whole players in one split (no player overlap, tested); `random_stratified` is available but is optimistic because the same player appears across splits. Class imbalance: positive weight `(neg/pos)^0.5` in every model; Platt calibration (validation) corrects the resulting probability inflation. For classical models sequences are aggregated (workload: 7 summaries + {mean28, std28, max28, mean7} per channel; wellness: {mean, std, last, slope} per variable).

## 6. Training and evaluation protocol
Validation data selects the NN checkpoint (early stopping), fits calibration and picks the decision threshold (`val_f1` by default; `fixed`, `val_youden`, `val_recall_target` available). The test set is used once for the reported metrics, and every model is scored on the same test records. Metrics: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, ROC/PR/calibration curves - all computed from the saved prediction files.

## 7. Explainability
- SHAP TreeExplainer on the XGBoost model (additivity tested).
- Integrated Gradients for the neural network: `IG_i = (x_i - x'_i) * mean_alpha dF/dx_i`, baseline = training mean (zero after scaling), `F` = logit; completeness error is reported.
- Permutation importance at modality and feature level; SHAP GradientExplainer on the NN as a cross-check.
These describe model behaviour, not causal mechanisms.
