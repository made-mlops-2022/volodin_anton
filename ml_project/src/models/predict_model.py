import sklearn.linear_model
import sklearn.ensemble
import hydra
import logging
import pandas as pd
import pickle
from configs.config import PredictParams


@hydra.main(
    version_base=None, config_path="../../configs", config_name="predict_config"
)
def predict_model(cfg: PredictParams):
    logging.info("Reading source data")
    X = pd.read_csv(cfg.input_data_path)

    logging.info("Loading model")
    with open(cfg.model_path, "rb") as f:
        model = pickle.load(f)

    logging.info("Predicting target")
    prediction = model.predict(X)
    prediction = pd.DataFrame(prediction)

    logging.info(f"Saving prediction to {cfg.save_path}")
    prediction.to_csv(cfg.save_path, index=False)
