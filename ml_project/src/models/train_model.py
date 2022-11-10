import hydra
from configs.config import TrainParams
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sklearn.linear_model
import sklearn.ensemble
from src.features import build_features


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_config_1"
)
def train_model(cfg: TrainParams):
    X, y = build_features.split_to_features_and_target(
        cfg.input_data_path, cfg.target_name
    )
    model = eval(cfg.model)()
    logging.info("Splitting data to train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.splitting_params.test_size,
        random_state=cfg.splitting_params.random_state,
    )

    logging.info("Fitting model")
    model.fit(X_train, y_train)

    logging.info("Predicting test")
    prediction = model.predict(X_test)
    metric = f1_score(y_test, prediction)
    logging.info(f"F1-score is {metric}")

    logging.info("Dumping model")
    with open(cfg.dump_model, "wb") as handler:
        pickle.dump(model, handler)


if __name__ == "__main__":
    train_model()
