import os
import pandas as pd
import pickle
import numpy as np
import click
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
         ("scaler", StandardScaler()),
        ]
    )
    return num_pipeline


def build_transformer() -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                NUMERICAL_FEATURES,
            ),
        ]
    )
    return transformer


def create_inference_pipeline(model: LogisticRegression, transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("process_feature_part", transformer), ("model_part", model)])


def train_model(data) -> Pipeline:
    transformer = build_transformer()
    y = data['condition']
    X = data.drop('condition', axis=1, inplace=False)
    transformer.fit(X)
    transform_train_df = transformer.transform(X)
    model = LogisticRegression(
        C=0.1, 
        random_state=42,
    )
    model.fit(transform_train_df, y)
    inference_pipeline = create_inference_pipeline(model, transformer)
    return inference_pipeline


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def main(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))

    model = train_model(data)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
