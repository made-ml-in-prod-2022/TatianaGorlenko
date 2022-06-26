from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys

from ml_project.entities.predict_pipeline_params import read_prediction_pipeline_params
from ml_project.utils.make_dataset import read_data
from ml_project.models.model_fit_predict import load_model, predict_model, evaluate_model, save_metrics, save_prediction


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def setup_parser(parser):
    parser.add_argument('-c', '--config', required=True, help='path to predict config')
    parser.set_defaults(callback=callback_predict_model)


def callback_predict_model(arguments):
    run_predict(arguments.config)


def run_predict(conf_path):
    predict_pipeline_params = read_prediction_pipeline_params(conf_path)
    logger.info(f"Start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    logger.info(f"Loading model from {predict_pipeline_params.model_path}...")
    inference_pipeline = load_model(predict_pipeline_params.model_path)
    logger.info(f"Start predict...")
    predict = predict_model(inference_pipeline, data)
    pred_path = save_prediction(predict_pipeline_params.prediction_path, predict)
    logger.info(f"Prediction saved in {pred_path}")
    if predict_pipeline_params.target_path:
        logger.info(f"Compute score...")
        target = read_data(predict_pipeline_params.target_path)
        metrics = evaluate_model(predict, target)
        save_metrics(metrics, predict_pipeline_params.metric_path)
        logger.info(f"Мetrics is {metrics}")
    else:
        logger.info(f"There is no test_target_file")
    logger.info(f"End predict pipeline")
    return pred_path


def main():
    parser = ArgumentParser(
        prog='predict model',
        description='predict model',
        )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)

if __name__ == "__main__":
    main()
