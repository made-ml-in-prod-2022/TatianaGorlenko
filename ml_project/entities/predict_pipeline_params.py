#from typing import Optional

from dataclasses import dataclass

# from .download_params import DownloadParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictionPipelineParams:
    input_data_path: str
    prediction_path: str
    target_path : str
    model_path: str
    metric_path: str

    # downloading_params: Optional[DownloadParams] = None
    # use_mlflow: bool = True
    # mlflow_uri: str = "http://18.156.5.226/"
    # mlflow_experiment: str = "inference_demo"


PredictionPipelineParamsSchema = class_schema(PredictionPipelineParams)


def read_prediction_pipeline_params(path: str) -> PredictionPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictionPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
