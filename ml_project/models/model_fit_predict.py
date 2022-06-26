import json
import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline

from ml_project.entities.train_params import TrainingParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators, 
            random_state=train_params.random_state,
            criterion=train_params.criterion,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=train_params.C, 
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def save_model(model: object, output: str) -> str:
    '''Save model in output file'''
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(input: str) -> object:
    '''Load model from input file'''
    with open(input, 'rb') as f:
        model = pickle.load(f)
    return model


def create_inference_pipeline(model: SklearnClassifierModel, transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("process_feature_part", transformer), ("model_part", model)])


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "f1_score": f1_score(target, predicts),
        "recall": recall_score(target, predicts),
        "precision": precision_score(target, predicts),
    }

def save_metrics(metrics: Dict[str, float], output: str) -> None:
    with open(output, "w") as f:
        json.dump(metrics, f)


def save_prediction(output: str, prediction: np.ndarray) ->  str:
    with open(output, mode='w') as f:
            f.write(str(prediction))
    return output
        