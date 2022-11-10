import os.path
import unittest
from configs.config import TrainParams, SplittingParams
import pandas as pd
from src.models.train_model import train_model


class TestTrain(unittest.TestCase):
    def test_train(self):
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
        cfg = TrainParams(
            input_data_path="data/raw/heart_cleveland_upload.csv",
            target_name="condition",
            splitting_params=SplittingParams(test_size=0.2, random_state=47),
            model="sklearn.linear_model.LogisticRegression",
            dump_model="models/test.pkl",
        )
        train_model(cfg)
        self.assertTrue(os.path.exists("models/test.pkl"))


if __name__ == "__main__":
    unittest.main()
