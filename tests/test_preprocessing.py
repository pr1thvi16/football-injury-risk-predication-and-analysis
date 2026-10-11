import copy

import numpy as np
import pytest

from src.config import load_config
from src.data.preprocessing import Preprocessor, compute_pos_weight, describe_splits, extract_target, make_splits
from src.features.feature_engineering import TARGET, sequences_from_frame


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_paper_split_sizes_grouped(full_df, cfg):
    sp = make_splits(full_df, cfg)
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (8160, 1440, 2400)
    info = describe_splits(full_df, sp, cfg)
    assert all(v == 0 for v in info["record_overlap"].values())
    assert all(v == 0 for v in info["player_overlap"].values())      # unseen players in val/test
    for k in sp:
        assert abs(info["splits"][k]["prevalence"] - 0.0898) < 0.01


def test_random_stratified_option(full_df, cfg):
    c = copy.deepcopy(cfg); c["split"]["strategy"] = "random_stratified"
    sp = make_splits(full_df, c)
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (8160, 1440, 2400)
    assert not set(sp["train"]) & set(sp["test"])
    info = describe_splits(full_df, sp, c)
    assert info["player_overlap"]["train&test"] > 0       # documented: same players appear in several splits


def test_split_is_deterministic(full_df, cfg):
    a, b = make_splits(full_df, cfg), make_splits(full_df, cfg)
    assert all((a[k] == b[k]).all() for k in a)


def test_unknown_strategy_raises(full_df, cfg):
    c = copy.deepcopy(cfg); c["split"]["strategy"] = "nope"
    with pytest.raises(ValueError):
        make_splits(full_df, c)


def test_fit_on_train_only_and_shapes(small_df, cfg):
    c = copy.deepcopy(cfg)
    sp = make_splits(small_df, c)
    tr, te = small_df.iloc[sp["train"]], small_df.iloc[sp["test"]]
    pre = Preprocessor().fit(tr)
    a_tr, a_te = pre.transform(tr), pre.transform(te)
    n = len(te)
    assert a_te["static"].shape == (n, 10 + 6) and a_te["wl_seq"].shape == (n, 28, 8)
    assert a_te["wb_seq"].shape == (n, 7, 7) and a_te["wl_sum"].shape == (n, 7)
    wl_tr, _ = sequences_from_frame(tr)
    assert np.allclose(pre.wl_mean, wl_tr.mean(axis=(0, 1)))          # statistics come from TRAIN only
    assert np.allclose(a_tr["wl_seq"].mean(axis=(0, 1)), 0, atol=1e-4)
    assert abs(a_te["wl_seq"].mean(axis=(0, 1))).max() > 1e-4          # test is transformed with train stats, not re-centred
    pre_all = Preprocessor().fit(small_df)
    assert not np.allclose(pre.wl_mean, pre_all.wl_mean)              # fitting on all data would differ (leak)


def test_preprocessor_never_uses_target(small_df, cfg):
    flipped = small_df.copy(); flipped[TARGET] = 1 - flipped[TARGET]
    a, b = Preprocessor().fit(small_df), Preprocessor().fit(flipped)
    assert np.allclose(a.transform(small_df)["tab_workload"], b.transform(small_df)["tab_workload"])


def test_missing_values_are_imputed_without_nan(small_df, cfg):
    pre = Preprocessor().fit(small_df)
    d = small_df.head(5).copy()
    d.loc[d.index[0], "age"] = np.nan
    d.loc[d.index[1], "wl_total_distance_km_d4"] = np.nan
    d.loc[d.index[2], "wb_sleep_hours_d2"] = np.nan
    out = pre.transform(d)
    assert all(np.isfinite(v).all() for v in out.values())


def test_target_extraction_and_pos_weight(small_df):
    y = extract_target(small_df)
    assert set(np.unique(y)) <= {0, 1}
    assert compute_pos_weight(np.array([0] * 90 + [1] * 10), 0.5) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        extract_target(small_df.drop(columns=[TARGET]))


def test_preprocessor_save_load_and_errors(small_df, tmp_path):
    pre = Preprocessor().fit(small_df)
    pre.save(tmp_path / "p.joblib")
    again = Preprocessor.load(tmp_path / "p.joblib")
    assert np.allclose(pre.transform(small_df.head(3))["static"], again.transform(small_df.head(3))["static"])
    with pytest.raises(FileNotFoundError):
        Preprocessor.load(tmp_path / "missing.joblib")
    (tmp_path / "bad.joblib").write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="corrupted"):
        Preprocessor.load(tmp_path / "bad.joblib")
