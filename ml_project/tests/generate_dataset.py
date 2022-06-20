import numpy as np
import pandas as pd


def generate_dataset(count_objects: int, is_train: bool = True) -> pd.DataFrame:
    np.random.seed(123)
    dataset = pd.DataFrame()
    target = [1] * count_objects
    target[0] = 0
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
    if is_train: 
        dataset["condition"] = np.array(target)

    return dataset
