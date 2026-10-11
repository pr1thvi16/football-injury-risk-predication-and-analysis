"""Export result JSON for the website. Everything is derived from saved predictions/results (SYNTHETIC unless stated)."""
import json
from pathlib import Path

import pandas as pd

from src.paper.model import PaperModel
from src.paper.run import load_cfg


def main():
    cfg = load_cfg(); r = Path(cfg["paths"]["results_dir"]); out = Path("results/web"); out.mkdir(parents=True, exist_ok=True)
    tag = "static_load_wellness"
    comp = pd.read_csv(r / f"model_comparison_{tag}.csv")
    extra = json.load(open(r / f"extra_{tag}.json"))
    res = {"data_source": "SYNTHETIC", "note": "Simulated data; not clinical evidence.", "models": comp.round(4).to_dict("records"),
           "calibration": pd.read_csv(r / f"calibration_{tag}.csv").round(4).to_dict("records"),
           "ablation": pd.read_csv(r / f"ablation_{tag}.csv").round(4).to_dict("records"),
           "subgroups": pd.read_csv(r / f"subgroups_{tag}.csv").round(4).to_dict("records"), **extra}
    json.dump(res, open(out / "research-results.json", "w"), indent=2)
    json.dump({"data_source": "SYNTHETIC", "unit": "probability (permutation Shapley)",
               "features": pd.read_csv(r / "shapley_global.csv").round(5).to_dict("records"),
               "modality_share": pd.read_csv(r / "shapley_modality_share.csv").round(2).to_dict("records")},
              open(out / "feature-importance.json", "w"), indent=2)
    m = PaperModel.load(Path(cfg["paths"]["models_dir"]) / f"fusion_{tag}.joblib")
    test = pd.read_csv(r / "split_test.csv"); test["risk"] = m.predict_proba(test).round(4)
    last = test.sort_values("observation_date").groupby("player_id").tail(1).sort_values("risk", ascending=False)
    keep = ["player_id", "observation_date", "position", "age", "acwr", "risk", "injury_next_7_days"]
    json.dump({"data_source": "SYNTHETIC", "players": last[keep].round(3).to_dict("records")}, open(out / "players.json", "w"), indent=2)
    trend = test.groupby("week_index").agg(mean_risk=("risk", "mean"), mean_acwr=("acwr", "mean"), injuries=("injury_next_7_days", "sum")).reset_index()
    json.dump({"data_source": "SYNTHETIC", "weeks": trend.round(4).to_dict("records")}, open(out / "squad-trends.json", "w"), indent=2)
    print("wrote", [p.name for p in out.glob("*.json")])


if __name__ == "__main__":
    main()
