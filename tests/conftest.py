import pytest

from src.config import load_config
from src.data.generate_dataset import generate_dataset


@pytest.fixture(scope="session")
def full_df():
    """The full 12,000-record SYNTHETIC dataset (generated in memory, default seed)."""
    return generate_dataset()


@pytest.fixture(scope="session")
def small_df():
    return generate_dataset(n_players=60, n_weeks=16, seed=7)


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    return load_config(overrides={
        "data": {"n_players": 60, "n_weeks": 16},
        "paths": {"dataset": str(root / "data.csv"), "sample_dataset": str(root / "sample.csv"),
                  "models_dir": str(root / "models"), "metrics_dir": str(root / "metrics"),
                  "predictions_dir": str(root / "preds"), "figures_dir": str(root / "figs"),
                  "explain_dir": str(root / "explain")},
        "nn_training": {"max_epochs": 3, "patience": 3},
    })


@pytest.fixture(scope="session")
def trained(small_cfg, small_df):
    """Run the real training pipeline (smoke settings) on a small synthetic dataset."""
    from src.training.train import run_training
    small_df.to_csv(small_cfg["paths"]["dataset"], index=False)
    out = run_training(small_cfg, smoke=True, verbose=False)
    return small_cfg, out
