# Paper-faithful pipeline (`src/paper`) - results and honest comparison

All numbers below are from `python -m src.paper.run train` on the **SYNTHETIC** dataset (12,000 player-weeks, 1,078 positives = 8.98 %, player-grouped split 8,160 / 1,440 / 2,400; test = 264 positives). Single seed; 95 % CIs from 1,000 test-set bootstrap resamples. Files: `results/paper/`.

| Model | ROC-AUC (95 % CI) | PR-AUC | Paper AUC |
|---|---|---|---|
| Fusion (3 MLP encoders + LR head) | 0.713 (0.681-0.744) | 0.300 | 0.706 |
| LR static only | 0.687 (0.653-0.720) | 0.299 | 0.693 |
| MLP load only | 0.499 (0.463-0.532) | 0.107 | - |
| MLP wellness only | 0.590 (0.554-0.623) | 0.159 | - |
| LR early fusion | 0.717 (0.686-0.749) | 0.324 | 0.719 |
| Random forest | 0.658 | 0.217 | - |
| Gradient boosting | 0.688 | 0.249 | - |
| MLP early fusion | 0.711 | 0.293 | - |

**Reading.** The fusion model beats static-only (paired bootstrap diff +0.026, 95 % CI 0.003-0.049) but is **not** better than early-fusion logistic regression (diff -0.004, CI -0.017 to +0.009). Static features dominate (Shapley modality share: static 50.8 %, wellness 29.8 %, workload 19.4 %; mean-substitution ablation drop: static 0.116, wellness 0.038, workload 0.002). Workload-only MLP is at chance on this synthetic mechanism. The closeness to the paper's quoted AUCs is a property of how this simulator was built (assumptions in `src/paper/generator.py`) and must not be read as reproducing the paper: the paper's data were also simulated and its generator is not published.

**Implementation notes.** sklearn's built-in `early_stopping` monitors accuracy, which is flat at 9 % prevalence and stopped training after ~2 epochs (wellness-only AUC 0.48). We therefore early-stop on held-out log-loss (player-grouped inner split) and restore the best weights. Fusion-head CV (5-fold) is optimistic because encoders saw those rows.
