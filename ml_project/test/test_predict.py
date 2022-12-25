import os.path
import unittest
import pandas as pd

from configs.config import MakeDatasetParams, PredictParams
from src.models.predict_model import predict_model
from src.data.make_dataset import make_dataset


class TestPredict(unittest.TestCase):
    def test_predict(self):
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
            name="dataset_for_test_test.csv",
            n_samples=30,
            add_target=False
        )
        make_dataset(dataset_cfg)

        cfg = PredictParams(
            input_data_path=dataset_cfg.save_path + dataset_cfg.name,
            model_path="models/test.pkl",
            save_path="models/test_prediction.csv",
        )
        predict_model(cfg)
        self.assertTrue(os.path.exists("models/test_prediction.csv"))


if __name__ == "__main__":
    unittest.main()
