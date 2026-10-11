import numpy as np
import pandas as pd
import pytest

from src.paper import evaluation as ev
from src.paper import shapley
from src.paper.features import LOAD_FEATURES, TARGET, WELLNESS_FEATURES
from src.paper.generator import MAPPING, generate_tables
from src.paper.model import PaperModel
from src.paper.real_data import MappingError, build_player_weeks


@pytest.fixture(scope="module")
def tables():
    return generate_tables(n_players=40, n_weeks=20, prevalence=0.0898, seed=1)


@pytest.fixture(scope="module")
def weeks(tables):
    return build_player_weeks(*tables, MAPPING)


def test_exact_positive_count(weeks):
    assert len(weeks) == 40 * 20 and weeks[TARGET].sum() == round(0.0898 * len(weeks))


def test_features_do_not_see_the_future(tables):
    daily, inj, static = tables
    base = build_player_weeks(daily, inj, static, MAPPING)
    d2 = daily.copy()
    d2["date"] = pd.to_datetime(d2["date"])
    cutoff = pd.Timestamp(base.observation_date.iloc[3])
    first = base.player_id.iloc[3]
    mask = (d2.player_id == first) & (d2.date >= cutoff)
    d2.loc[mask, "session_load"] = 99999
    d2["date"] = d2["date"].dt.strftime("%Y-%m-%d")
    changed = build_player_weeks(d2, inj, static, MAPPING)
    assert np.allclose(base.loc[3, LOAD_FEATURES + WELLNESS_FEATURES].astype(float), changed.loc[3, LOAD_FEATURES + WELLNESS_FEATURES].astype(float))


def test_round_trip_through_column_mapping(tables, weeks):
    daily, inj, static = tables
    ren = {"player_id": "pid", "date": "day", "session_load": "ld", "minutes_played": "mins",
           "fatigue": "f", "sleep": "s", "soreness": "so", "mood": "m"}
    m = {"daily": {"player_id": "pid", "date": "day", "load": "ld", "minutes_played": "mins",
                   "wellness": {"fatigue": "f", "sleep": "s", "soreness": "so", "mood": "m"}},
         "injuries": {"player_id": "p", "date": "d"}, "static": MAPPING["static"], "options": {"data_source": "REAL"}}
    out = build_player_weeks(daily.rename(columns=ren), inj.rename(columns={"player_id": "p", "injury_date": "d"}), static, m)
    cols = LOAD_FEATURES + WELLNESS_FEATURES + [TARGET]
    pd.testing.assert_frame_equal(out[cols].reset_index(drop=True), weeks[cols].reset_index(drop=True))
    assert (out.data_source == "REAL").all()


def test_bad_mapping_is_reported(tables):
    daily, inj, static = tables
    bad = {**MAPPING, "daily": {**MAPPING["daily"], "load": "nope"}}
    with pytest.raises(MappingError):
        build_player_weeks(daily, inj, static, bad)


def test_grouped_split_has_no_player_overlap(weeks):
    s = ev.grouped_split(weeks, 68, 12, 20)
    sets = [set(v.player_id) for v in s.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(v) for v in s.values()) == len(weeks)


@pytest.fixture(scope="module")
def fitted(weeks):
    s = ev.grouped_split(weeks, 68, 12, 20)
    return PaperModel(max_iter=40).fit(s["train"]), s


def test_model_predicts_valid_probabilities_and_embedding_dim(fitted):
    m, s = fitted
    p = m.predict_proba(s["test"])
    assert p.shape == (len(s["test"]),) and ((p >= 0) & (p <= 1)).all()
    assert m.encoders["load"].embed(s["test"]).shape[1] == 16


def test_fine_tune_returns_new_model_and_leaves_original(fitted):
    m, s = fitted
    before = m.predict_proba(s["test"]).copy()
    m2 = m.fine_tune(s["val"], max_iter=3)
    assert np.allclose(before, m.predict_proba(s["test"])) and m2 is not m


def test_load_and_wellness_only_model_works_without_static(fitted):
    _, s = fitted
    m = PaperModel(["load", "wellness"], max_iter=20).fit(s["train"].assign(age=np.nan, bmi=np.nan))
    assert np.isfinite(m.predict_proba(s["test"])).all()


def test_shapley_efficiency_approximately_holds(fitted):
    m, s = fitted
    row = s["test"].iloc[0]
    bg = s["train"].sample(30, random_state=0).reset_index(drop=True)
    phi = shapley.shapley_values(m.predict_proba, row, bg, m.raw_columns, 200, np.random.default_rng(0))
    gap = m.predict_proba(pd.DataFrame([row]))[0] - m.predict_proba(bg).mean()
    assert abs(phi.sum() - gap) < 0.08


def test_youden_threshold_between_0_and_1():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 500); p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    assert 0 < ev.youden_threshold(y, p) < 1
