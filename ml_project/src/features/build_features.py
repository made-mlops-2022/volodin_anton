import pandas as pd
import logging


def split_to_features_and_target(file_path: str, target: str):
    logging.info("Loading data and splitting to features and target")

    df = pd.read_csv(file_path)
    X, y = df.drop(target, axis=1), df[target]

    logging.info("Data loaded")
    return X, y
