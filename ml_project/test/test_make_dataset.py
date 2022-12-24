import unittest
import os

from src.data.make_dataset import make_dataset
from configs.config import MakeDatasetParams



class TestMakeDataset(unittest.TestCase):
    def test_make_dataset(self):
        cfg = MakeDatasetParams(
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
            name="test_generated_dataset.csv",
            n_samples=30,
            add_target=True,
        )
        make_dataset(cfg)
        self.assertTrue(os.path.exists("data/raw/test_generated_dataset.csv"))


if __name__ == "__main__":
    unittest.main()
