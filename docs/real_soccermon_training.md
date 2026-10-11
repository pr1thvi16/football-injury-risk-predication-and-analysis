# Training on the real SoccerMon dataset

The default pipeline in `src/training/train.py` intentionally trains on the repository's **synthetic** benchmark. Do not report those metrics as results from real football players.

This separate script downloads the public **subjective** SoccerMon archive from Zenodo at run time and trains baseline classifiers on player-reported workload/wellness and injury events:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_soccermon.py
```

The script stores its dataset and metrics under `results/real_soccermon/`. It does not download the very large GPS archive. The subjective archive is much smaller and includes training-load, wellness and injury reports.

## Important validation notes

- The script is a real-data baseline pipeline, not the existing synthetic multimodal neural network. It currently fits Logistic Regression and Random Forest.
- The target is a **player-reported injury event in the next 7 days**, not a clinically verified diagnosis.
- Inspect any schema warnings emitted by the script. The source archive has multiple report tables, and its exact columns must be verified before interpreting metrics.
- Check that injury event dates and feature report dates are aligned correctly. Do not present results until the schema review and label audit pass.
- It uses a chronological date split; player-days are not randomly shuffled across train/test.
- This public dataset represents two elite Norwegian women's teams in 2020–2021. It should not be assumed to generalize to men's football or other leagues.

Source: [SoccerMon Zenodo record](https://zenodo.org/records/10033832), licensed CC BY 4.0. Cite the dataset paper: [Scientific Data (2024)](https://doi.org/10.1038/s41597-024-03386-x).
