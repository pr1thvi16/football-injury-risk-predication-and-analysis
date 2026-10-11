# Data-quality report (SYNTHETIC dataset)

- Records: **12,000**  |  Players: **250**
- Positives: **1,078** (8.98%)
- Date range: 2023-08-07 -> 2024-07-01
- Overall: **PASS**

| Check | Result | Detail |
|---|---|---|
| schema: all required columns present | PASS | missing=[] |
| record count == 12000 | PASS | n=12000 |
| observation_id unique | PASS |  |
| player_id valid (P###) | PASS |  |
| no missing values | PASS | missing cells=0 |
| no duplicate records | PASS | duplicates=0 |
| no duplicate (player, week) pairs | PASS |  |
| no impossible negative measurements | PASS | cols=[] |
| position values valid | PASS | unique=['CB', 'CM', 'FB', 'GK', 'ST', 'W'] |
| wellness scores within documented scales | PASS | bad=[] |
| target encoded as {0,1} | PASS | values=[0, 1] |
| target prevalence ~ 8.98% | PASS | prevalence=8.9833%, positives=1078 |
| observation dates valid | PASS | 2023-08-07 -> 2024-07-01 |
| data_source == SYNTHETIC for every row | PASS |  |
| derived workload features match their documented formulas | PASS |  |
| session_training_load == session_rpe * duration | PASS |  |
| feature/target separation (no target-like column in model inputs) | PASS | suspicious=[] |
