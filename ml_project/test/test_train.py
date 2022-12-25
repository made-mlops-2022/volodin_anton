import os.path
import unittest
import pandas as pd

from configs.config import MakeDatasetParams, TrainParams, SplittingParams
from src.models.train_model import train_model
from src.data.make_dataset import make_dataset


class TestTrain(unittest.TestCase):
    def test_train(self):
        dataset_cfg = MakeDatasetParams(
            input_data_path="data/raw/heart_cleveland_upload.csv",
            target_name="condition",
            features={
                "categorical": [
                    "sex",
                    "cp",
                    "fbs",
                    "restecg",
                    "exang",
                    "slope",
                    "ca",
                    "thal",
                ],
                "numerical": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            },
            save_path="data/raw/",
            name="dataset_for_train_test.csv",
            n_samples=30,
            add_target=True,
        )
        make_dataset(dataset_cfg)

        cfg = TrainParams(
            input_data_path=dataset_cfg.save_path + dataset_cfg.name,
            target_name="condition",
            splitting_params=SplittingParams(test_size=0.2, random_state=47),
            model="sklearn.linear_model.LogisticRegression",
            dump_model="models/test.pkl",
        )
        train_model(cfg)
        self.assertTrue(os.path.exists("models/test.pkl"))


if __name__ == "__main__":
    unittest.main()
