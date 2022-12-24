import pandas as pd
import hydra
import numpy as np
import logging

from src.features import build_features
from configs.config import MakeDatasetParams


@hydra.main(version_base=None, config_path="../../configs", config_name="make_dataset")
def make_dataset(cfg: MakeDatasetParams):
    X, y = build_features.split_to_features_and_target(
        cfg.input_data_path, cfg.target_name
    )

    logging.info("Generating data")
    generated = {}
    for col in X.columns:
        if col in cfg.features["numerical"]:
            values = np.random.uniform(X[col].min(), X[col].max(), cfg.n_samples)
        else:
            values = np.random.choice(X[col].values, cfg.n_samples)
        generated[col] = values

    if cfg.add_target:
        generated[y.name] = np.random.randint(2, size=cfg.n_samples)

    generated = pd.DataFrame(generated)

    logging.info(f"Saving data to {cfg.save_path}")
    generated.to_csv(cfg.save_path + cfg.name, index=False)
