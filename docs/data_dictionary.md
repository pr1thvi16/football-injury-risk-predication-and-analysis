# Data dictionary

All values are **SYNTHETIC**. See `docs/methodology.md` for how they are simulated.

| Feature | Group | Type | Unit | Valid range | Direction | Meaning |
|---|---|---|---|---|---|---|
| `observation_id` | Identifier | string | - | unique | - | Unique id of the player-week record. |
| `player_id` | Identifier | string | - | P001-P250 | - | Stable synthetic player identifier. |
| `week_index` | Identifier | int | week | 0-47 | - | Index of the weekly assessment. |
| `observation_date` | Identifier | date | - | 2023-08-07 onward | - | Assessment date (Monday). History windows end the day before; the target covers the next 7 days. |
| `data_source` | Identifier | string | - | SYNTHETIC | - | Always SYNTHETIC - not real club data. |
| `age` | Static & medical | int | years | 17-40 | - | Player age. |
| `position` | Static & medical | category | - | GK, CB, FB, CM, W, ST | - | Playing position (goalkeeper, centre-back, full-back, central midfielder, winger, striker). |
| `height_cm` | Static & medical | float | cm | 155-210 | - | Height. |
| `weight_kg` | Static & medical | float | kg | 55-110 | - | Body mass. |
| `bmi` | Static & medical | float | kg/m^2 | 17-30 | - | weight_kg / (height_cm/100)^2. |
| `previous_injury_count` | Static & medical | int | count | 0-10 | - | Number of previous time-loss injuries. |
| `previous_injury_days_missed` | Static & medical | int | days | 0-600 | - | Total days missed through previous injuries. |
| `days_since_last_injury` | Static & medical | int | days | 0-730 | - | Days since the last injury; capped at 730, players without injury = 730. |
| `season_minutes_played` | Static & medical | int | minutes | 0-8000 | - | Cumulative match minutes up to the observation date (simulation period). |
| `strength_asymmetry_percent` | Static & medical | float | % | 0-40 | higher = worse | Inter-limb strength asymmetry. Higher = more asymmetric. |
| `average_rest_days` | Static & medical | float | days/week | 0-7 | - | Mean rest days per week over the 28-day window (days with zero training). |
| `acute_workload` | Workload summary (derived) | float | AU/day | >= 0 | - | mean session load, last 7 days |
| `chronic_workload` | Workload summary (derived) | float | AU/day | >= 0 | - | mean session load, last 28 days |
| `acute_chronic_workload_ratio` | Workload summary (derived) | float | ratio | >= 0 | - | acute / max(chronic, 1). A candidate feature only, not a definitive predictor. |
| `workload_monotony` | Workload summary (derived) | float | ratio | >= 0 | - | mean_7d / std_7d, capped at 10 |
| `workload_strain` | Workload summary (derived) | float | AU | >= 0 | - | sum of load over 7 days x monotony |
| `workload_variability` | Workload summary (derived) | float | ratio | >= 0 | - | coefficient of variation of 28-day load: std_28d / max(chronic, 1) |
| `recent_workload_change` | Workload summary (derived) | float | ratio | >= 0 | - | (acute - mean of days 15-21 back) / max(that mean, 1) |
| `wl_total_distance_km_d1 ... d28` | Workload series | float | km | 0-20 | - | Total distance covered per day (training + match). Columns d1 (oldest) to d28 (day before observation). |
| `wl_high_speed_running_distance_km_d1 ... d28` | Workload series | float | km | 0-5 | - | Distance above the high-speed threshold per day. Columns d1 (oldest) to d28 (day before observation). |
| `wl_sprint_distance_km_d1 ... d28` | Workload series | float | km | 0-2 | - | Sprint distance per day. Columns d1 (oldest) to d28 (day before observation). |
| `wl_acceleration_count_d1 ... d28` | Workload series | float | count | 0-140 | - | Number of accelerations per day. Columns d1 (oldest) to d28 (day before observation). |
| `wl_deceleration_count_d1 ... d28` | Workload series | float | count | 0-150 | - | Number of decelerations per day. Columns d1 (oldest) to d28 (day before observation). |
| `wl_training_duration_minutes_d1 ... d28` | Workload series | float | min | 0-130 | - | Session duration per day (0 = rest day). Columns d1 (oldest) to d28 (day before observation). |
| `wl_session_rating_of_perceived_exertion_d1 ... d28` | Workload series | float | CR-10 scale | 0-10 | - | Session RPE; 0 only on rest days. Columns d1 (oldest) to d28 (day before observation). |
| `wl_session_training_load_d1 ... d28` | Workload series | float | AU | 0-1300 | - | session_rpe x duration_minutes. Columns d1 (oldest) to d28 (day before observation). |
| `wb_sleep_hours_d1 ... d7` | Wellness series | float | hours | 3.5-10.5 | higher = better | Night sleep duration. Columns d1 (oldest) to d7 (day before observation). |
| `wb_sleep_quality_d1 ... d7` | Wellness series | float | 1-10 | 1-10 | higher = better | Self-rated sleep quality. Columns d1 (oldest) to d7 (day before observation). |
| `wb_fatigue_score_d1 ... d7` | Wellness series | float | 1-10 | 1-10 | higher = worse | Self-rated fatigue. Columns d1 (oldest) to d7 (day before observation). |
| `wb_muscle_soreness_score_d1 ... d7` | Wellness series | float | 1-10 | 1-10 | higher = worse | Self-rated muscle soreness. Columns d1 (oldest) to d7 (day before observation). |
| `wb_mood_score_d1 ... d7` | Wellness series | float | 1-10 | 1-10 | higher = better | Self-rated mood. Columns d1 (oldest) to d7 (day before observation). |
| `wb_recovery_score_d1 ... d7` | Wellness series | float | 0-100 | 0-100 | higher = better | Self-rated overall recovery. Columns d1 (oldest) to d7 (day before observation). |
| `wb_perceived_stress_score_d1 ... d7` | Wellness series | float | 1-10 | 1-10 | higher = worse | Self-rated stress. Columns d1 (oldest) to d7 (day before observation). |
| `injury_next_7_days` | Target | int | 0/1 | {0, 1} | - | 1 = a simulated time-loss injury occurs in the 7 days starting on the observation date. |
