# Limitations

1. **Synthetic data.** Everything is simulated; no real club data. Metrics describe how well models recover the simulator's mechanism, not real-world injury prediction.
2. **Association is not causation.** Feature importances (SHAP, Integrated Gradients, permutation) reflect the simulator's label rule and the models' behaviour. They do not show why a real injury occurs.
3. **Class imbalance.** About 9 % positives; the 2,400-record test set holds 216 positives, so metric differences of a few points between models are within sampling uncertainty. Accuracy is misleading here and is not emphasised.
4. **Generalisation.** One simulated cohort, one seed. The grouped split tests unseen *simulated* players only. A random record-level split (option provided) is optimistic because players repeat.
5. **Probability and thresholds.** Raw probabilities are inflated by class weighting; Platt calibration on a small validation set (about 130 positives) is itself noisy. Risk categories (20/40/60 %) are illustrative, not clinically validated.
6. **Model comparison.** In this run the multimodal network did not clearly outperform the classical baselines (see README). Hyper-parameters were fixed, not tuned.
7. **Method disagreement.** Different attribution methods give different modality shares; none is ground truth.
8. **Not a medical device.** No diagnosis, treatment or return-to-play use. Any real use needs prospective external validation on real data, ethics approval and clinical oversight.
