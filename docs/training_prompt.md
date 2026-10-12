# Prompt: train all models for this project

```
You are an ML engineer working in the repo football-injury-risk-prediction (macOS, CPU only, Python 3.11+).
Goal: train, evaluate and explain ALL models for 7-day football injury-risk prediction, first on the SYNTHETIC
dataset, then on any REAL dataset I provide. Never fabricate numbers: every metric must come from real predictions
on held-out data. Label synthetic results "SYNTHETIC".

Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install pytest`

1. Data: `python -m src.paper.run generate` (12,000 player-weeks, 1,078 positives = 8.98 %). Check no NaNs,
   exact positive count, features use only days BEFORE the observation date.
2. Split by PLAYER (no player in two sets): 8,160 / 1,440 / 2,400 (train/val/test). Fit every imputer, scaler and
   encoder on train only. Tune only on validation; touch test once.
3. Models (same split, same metrics): per-modality MLP encoders (static, 28-day workload, 7-day wellness,
   16-d embeddings, L2, early stopping on validation LOG-LOSS) + logistic-regression fusion head (C=0.05);
   baselines: LR static-only, MLP load-only, MLP wellness-only, LR early fusion, Random forest (400 trees,
   min leaf 5), Gradient boosting, MLP early fusion. Optional extension: PyTorch GRU+attention multimodal net
   (`python -m src.training.train`).
   Command: `python -m src.paper.run train` (all three modalities) and
   `python -m src.paper.run train --modalities load,wellness` (for data without static fields).
4. Evaluate on test: ROC-AUC with 1,000-resample bootstrap 95 % CI, PR-AUC, precision, recall, F1, Brier,
   threshold from Youden's J on validation, calibration by quintile, paired bootstrap of fusion vs each baseline,
   modality ablation, subgroup AUC (position, age), 5-fold CV of the head.
5. Explain: `python -m src.paper.run explain` (permutation Shapley, modality importance %). Report disagreements.
6. Real data: put files in data/real/ (git-ignored), copy config/real_data_mapping.example.yaml, set column names,
   scales and inverted items (check direction with correlations against prior load). Run
   `python -m src.paper.run real --mapping config/real_data_mapping.yaml`; for few injuries (<~100 positives) use
   `python -m src.paper.real_cv` (leave-players-out CV, player-cluster bootstrap CI). Compare zero-shot,
   fine-tuned and retrained models.
7. Report honestly: show confidence intervals, number of positives, where fusion does NOT beat simple baselines,
   and that a synthetic-trained model encodes the simulator's assumptions. Run `pytest` (58 tests) before
   committing. Do not commit raw real data.
```
