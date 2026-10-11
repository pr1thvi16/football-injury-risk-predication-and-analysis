# Real-data training (SoccerMon)

This workflow is separate from the synthetic benchmark. It automatically fetches the public SoccerMon subjective archive from Zenodo and trains baseline models.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_real_soccermon.py
```

The script retrieves only `subjective.zip` (about 706 kB), not the approximately 99 GB objective GPS archives. Internet access is needed at runtime. The archive is held in memory and extracted into a temporary directory; parsed outputs and models are saved under `results/real_soccermon/`.

## Available modalities and limitations

The public subjective dataset contains training-load, wellness, injury, illness and match-performance reports for 50 players from two elite Norwegian women's teams in 2020–2021. It has 33,849 subjective reports including 306 injury records. It does not provide all fields described in the paper (e.g., player age/height/weight, strength asymmetry, complete clinical injury history or sleep wearable streams). The large GPS archive is separate and must be added explicitly if GPS features are needed.

This script is an automated baseline adapter, not a claim that every modality in the paper is present. It saves `source_schema.json` so the dataset mapping can be reviewed. If the real archive schema does not map to player/date/numeric features and injury events, it stops rather than generating fake labels. Verify the mapped injury event dates and features before reporting metrics.

Labels are based on player-reported injury events, not medically verified time-loss injuries. The dataset is small and injury events are sparse; use chronological evaluation and describe uncertainty. Results will not necessarily match the paper's simulated-data metrics.

Dataset: https://zenodo.org/records/10033832
Dataset paper: https://doi.org/10.1038/s41597-024-03386-x
License: CC BY 4.0; cite the dataset paper and dataset DOI in your report.
