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
