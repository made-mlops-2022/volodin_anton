import os.path
import unittest
from configs.config import PredictParams
import pandas as pd
from src.models.predict_model import predict_model


class TestPredict(unittest.TestCase):
    def test_predict(self):
        data = [
            [46, 1, 3, 172, 210, 0, 2, 194, 0, 1, 0, 0, 1],
            [54, 1, 2, 127, 227, 1, 0, 176, 0, 4, 1, 3, 2],
            [47, 1, 2, 123, 353, 0, 2, 104, 0, 6, 1, 0, 0],
            [70, 1, 3, 105, 367, 0, 0, 143, 0, 2, 0, 0, 2],
            [31, 1, 3, 168, 502, 0, 0, 167, 0, 2, 1, 1, 2],
        ]
        df = pd.DataFrame(data)
        df.columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        cfg = PredictParams(
            input_data_path="data/raw/generated_dataset.csv",
            model_path="models/test.pkl",
            save_path="models/test_prediction.csv",
        )
        predict_model(cfg)
        self.assertTrue(os.path.exists("models/test_prediction.csv"))


if __name__ == "__main__":
    unittest.main()
