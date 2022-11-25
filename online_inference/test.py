from fastapi.testclient import TestClient
from app import app
import pandas as pd
import numpy as np


EXPECTED_TARGETS = [0, 1, 1, 0, 1]


def test_predict():
    with TestClient(app) as client:
        data = pd.read_csv("generated_dataset.csv")
        features = list(data.columns)
        predictions = []
        for i in range(5):
            request_data = [
                x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
            ]
            response = client.post(
                "/predict",
                json={"data": [request_data], "features": features},
            )
            assert response.status_code == 200
            predictions.append(response.json()[0]['condition'])
        assert predictions == EXPECTED_TARGETS
