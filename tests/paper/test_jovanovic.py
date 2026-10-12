from pathlib import Path

import pandas as pd
import pytest

from src.paper.jovanovic import prepare

RAW = Path("data/sample_external/jovanovic_raw.csv"); REF = Path("data/sample_external/jovanovic_processed.csv")


@pytest.mark.skipif(not (RAW.exists() and REF.exists()), reason="external sample data not downloaded (see docs/real_data_guide.md)")
def test_port_matches_original_r_output():
    mine = prepare(pd.read_csv(RAW)); ref = pd.read_csv(REF)
    m = ref.merge(mine, on=["Player.Name", "Date"], suffixes=("_ref", "_my"))
    assert len(m) == len(ref)
    for c in ["RPE.Load.Acute", "RPE.Load.Chronic", "RPE.Load.ACWR", "RPE.Load.ACWRrollPeak", "HS.Distance.ACWR"]:
        assert (m[c + ".Lag.0"] - m[c]).abs().max() < 1e-9
