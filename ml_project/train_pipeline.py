from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys

from ml_project.entities.train_pipeline_params import read_training_pipeline_params
from ml_project.utils.make_dataset import read_data, split_train_val_data
from ml_project.utils.preprocessing import extract_target, build_transformer, transform_features
from ml_project.models.model_fit_predict import train_model, save_model, create_inference_pipeline, predict_model, evaluate_model, save_metrics


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def setup_parser(parser):
    parser.add_argument('-c', '--config', required=True, help='path to config')
    parser.set_defaults(callback=callback_train_model)


def callback_train_model(arguments):
    run_train(arguments.config)


def run_train(conf_path):
    training_pipeline_params = read_training_pipeline_params(conf_path)
    logger.info(f"Start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    target = extract_target(data, training_pipeline_params.feature_params)
    data = data.drop(training_pipeline_params.feature_params.target_col, axis=1)
    train_df, val_df, train_target, val_target = split_train_val_data(
        data, target, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    logger.info(f"Start preprocessing data...")
    transformer.fit(train_df)
    transform_train_df = transform_features(transformer, train_df)
    logger.info(f"transform_train_df.shape is {transform_train_df.shape}")
    logger.info(f"Start fit model {training_pipeline_params.train_params.model_type}...")
    model = train_model(
        transform_train_df, train_target, training_pipeline_params.train_params
    )
    logger.info(f"Мodel successfully trained")
    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts = predict_model(inference_pipeline, val_df,)
    metrics = evaluate_model(predicts, val_target,)
    path_to_model = save_model(inference_pipeline, training_pipeline_params.output_model_path)
    save_metrics(metrics, training_pipeline_params.metric_path)
    logger.info(f"Мetrics is {metrics}")
    logger.info(f"End train pipeline")
    return path_to_model, metrics


def main():
    parser = ArgumentParser(
        prog='train model',
        description='train model',
        )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)

if __name__ == "__main__":
    main()
