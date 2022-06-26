import logging
import os
import pickle
from typing import List, Optional


import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import time


logger = logging.getLogger(__name__)
START_SLEEP_TIME = 20
LIVE_TIME = 120

class HeartDiseaseModel(BaseModel):
    data: List[List[int]]
    features: List[str]


class СonditionResponse(BaseModel):
    сondition: int


model: Optional[Pipeline] = None


def make_predict(data: List, features: List[str], model: Pipeline)-> List[СonditionResponse]:
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)

    return [
        СonditionResponse(сondition=int(сond)) for сond in predicts
    ]


app = FastAPI()
start_time = time.time()


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)
    time.sleep(START_SLEEP_TIME)
    


@app.get("/health")
def health() -> bool:
    if time.time() - start_time > LIVE_TIME:
        sys.exit(1)
    return not (model is None)


@app.post("/predict/", response_model=List[СonditionResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
