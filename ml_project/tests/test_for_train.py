import os
from ml_project.tests.generate_dataset import generate_dataset
from ml_project.train_pipeline import run_train



def test_run_train():
    dataset = generate_dataset(10)
    dataset.to_csv("ml_project/tests/for_tests/dataset.csv")
    path_to_model, metrics = run_train('ml_project/tests/config/config_for_test.yaml')
    assert os.path.exists(path_to_model)
    assert metrics['f1_score'] >= 0
    assert metrics['precision'] >= 0
    assert metrics['recall'] >= 0
    