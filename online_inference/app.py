import os
import pickle
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class Model(BaseModel):
    data: List[List[float]]
    features: List[str]


class Response(BaseModel):
    condition: int


model: Optional[Pipeline] = None
app = FastAPI()


@app.get("/")
def main():
    return "Main page"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL", default='model.pkl')
    model = load_object(model_path)


@app.get("/health")
def health() -> int:
    if model is None:
        return 400
    return 200


@app.post("/predict", response_model=List[Response])
def predict(request: Model):
    data = pd.DataFrame(request.data, columns=request.features)
    predict = model.predict(data)
    return [Response(condition=p) for p in predict]


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
