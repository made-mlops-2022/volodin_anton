import numpy as np
import pandas as pd
import requests


if __name__ == "__main__":
    data = pd.read_csv("generated_dataset.csv")
    request_features = list(data.columns)
    if requests.get("http://127.0.0.1:8000/health").ok:
        for i in range(25):
            request_data = [
                x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
            ]
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                json={"data": [request_data], "features": request_features},
            )

            if response.status_code == 200:
                print('Item to predict:', *[(f + '=' + str(round(d))) for f, d in zip(request_features, request_data)])
                print('Predicted label:', response.json()[0]['condition'])

