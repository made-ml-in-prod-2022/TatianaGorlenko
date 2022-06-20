import os
import pandas as pd
import pickle

import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--path-to-model")
@click.option("--output-dir")
def predict(input_dir: str, path_to_model: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    
    predicts = pd.DataFrame({"condition": model.predict(data)})

    os.makedirs(output_dir, exist_ok=True)
    predicts.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
    