import numpy as np
import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("./ml_project/data/test.csv")
    request_features = list(data.columns)
    for i in range(data.shape[0]):
        request_data = [int(x) for x in data.iloc[i].tolist()]
        print(request_data)
        response = requests.post(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())
        