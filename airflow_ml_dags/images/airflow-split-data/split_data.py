import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split_data(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))

    train, val = train_test_split(data, test_size=0.2, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)


if __name__ == '__main__':
    split_data()
