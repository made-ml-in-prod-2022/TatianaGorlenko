import os
import pandas as pd
import pickle
import click
import json
from sklearn.metrics import f1_score, recall_score, precision_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--models-dir")
@click.option("--output-dir")
def main(input_dir: str, models_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "val_data.csv"))
    with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)    

    X = data[[col_name for col_name in data.columns if col_name != "condition"]]
    target = data["condition"]
    predicts = model.predict(X)

    metrics = {
        "f1_score": f1_score(target, predicts),
        "recall": recall_score(target, predicts),
        "precision": precision_score(target, predicts),
    }    

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
