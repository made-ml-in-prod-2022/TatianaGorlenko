import os

import click
import numpy as np
import pandas as pd

#from sklearn.datasets import load_wine

def generate_dataset(count_objects: int, seed: int = 123) -> pd.DataFrame:
    np.random.seed(seed)
    dataset = pd.DataFrame()
    target = pd.DataFrame()
    target["condition"] = np.random.randint(2, size=count_objects)
    dataset["cp"] = np.random.randint(0, 3, size=count_objects)
    dataset["restecg"] = np.random.randint(0, 3, size=count_objects)
    dataset["slope"] = np.random.randint(0, 3, size=count_objects)
    dataset["ca"] = np.random.randint(0, 4, size=count_objects)
    dataset["thal"] = np.random.randint(0, 3, size=count_objects)
    dataset["sex"] = np.random.randint(0, 2, size=count_objects)
    dataset["fbs"] = np.random.randint(0, 2, size=count_objects)
    dataset["exang"] = np.random.randint(0, 2, size=count_objects)
    dataset["age"] = np.random.randint(29, 77, size=count_objects)
    dataset["trestbps"] = np.random.randint(90, 200, size=count_objects)
    dataset["chol"] = np.random.randint(120, 570, size=count_objects)
    dataset["thalach"] = np.random.randint(70, 205, size=count_objects)
    dataset["oldpeak"] = np.round(np.random.uniform(0, 7, size=count_objects), 1)
    return dataset, target

@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    X, y = generate_dataset(100)
    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)



if __name__ == '__main__':
    download()
    