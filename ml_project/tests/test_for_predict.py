import os
from ml_project.tests.generate_dataset import generate_dataset
from ml_project.train_pipeline import run_train
from ml_project.predict_pipeline import run_predict


def test_run_train():
    dataset = generate_dataset(10)
    dataset.to_csv("ml_project/tests/for_tests/dataset.csv")
    path_to_model, metrics = run_train('ml_project/tests/config/config_for_test.yaml')
    pred_path = run_predict('ml_project/tests/config/config_for_test_predict.yaml')
    assert os.path.exists(pred_path)
