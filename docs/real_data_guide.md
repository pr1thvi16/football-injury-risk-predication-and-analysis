# Using the pipeline with real data

The synthetic generator emits three tables (daily, injuries, static) and builds player-weeks through `src/paper/real_data.py`. Real data uses **the same builder**, so features and labels are defined identically (a test proves the column-mapping round trip).

1. Put your files in `data/real/` (git-ignored). Needed: one row per player-day (load or RPE x minutes, wellness items), one row per time-loss injury (player, date). Optional: player static table.
2. `cp config/real_data_mapping.example.yaml config/real_data_mapping.yaml`; edit column names, scales (`wellness_scales`), `invert`, `position_map`.
3. `python -m src.paper.run real --mapping config/real_data_mapping.yaml`

Three ways to use the synthetic-trained model are compared on the same player-grouped real test split, next to the baselines:
- `zero_shot` - synthetic model as is. Expect degradation: load units, wellness scales and injury definitions differ.
- `fine_tune` - warm-start encoders on the real training split and refit the head.
- `retrain` - train from scratch on real data.

Be realistic: a synthetic-trained model encodes the *simulator's* assumptions, not football physiology. Its value is the tested pipeline and a starting initialisation. Judge any real result by the test AUC CI and the number of test positives (the pipeline warns below 20 positives). Without a static table only load + wellness are used; a modality is dropped when fewer than 3 of its fields exist. SoccerMon-type data (1-5 scales, ~50 players, no static fields) fits this path but will have very few positives.

## SoccerMon result (REAL data, subjective release)
Converted with `python -m src.paper.soccermon` (injury episode = report with no report in the previous 14 days; an assumption, SoccerMon has no time-loss field), then `python -m src.paper.real_cv`. 50 players, 2,550 weekly player-weeks, only **30 injury-report episodes in 15 players**, giving **16 positive weeks (0.6 %) in 8 players** after the 28-day history requirement (5 positives for "major"-only). Leave-players-out CV, pooled AUC (player-cluster bootstrap 95 % CI): zero-shot synthetic model 0.62 (0.44-0.79), fine-tuned 0.48, retrained 0.51, LR 0.51, RF 0.58, GB 0.60, ACWR-only 0.46. **Every interval includes 0.5: this dataset cannot show that any model predicts injury.** Static/medical data is absent, so only load + wellness were used. Raw files are git-ignored (CC BY 4.0, cite Midoglu et al.).
