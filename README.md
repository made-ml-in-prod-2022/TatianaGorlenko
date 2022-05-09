# Homework 1

## Датасет
Для выполнения задания были взяты данные с сайта: https://www.kaggle.com/code/yuvrajgavhane/heart-disease-eda-and-modelling

Для тестовой выборки были выбраны случайные 10 строк и помещены в отдельную папку

## Модель
Доступно обучение моделей RandomForestClassifier и LogisticRegression

Запуск для RandomForestClassifier с конфигурацией:
~~~
python3 -m ml_project.train_pipeline -c ml_project/configs/config_random_forest.yaml
~~~

Запуск для LogisticRegression с конфигурацией:
~~~
python3 -m ml_project.train_pipeline -c ml_project/configs/config_logistic_regr.yaml
~~~

Сделать предсказание:
~~~
python3 -m ml_project.predict_pipeline -c ml_project/configs/config_predict.yaml
~~~

Запуск тестов:
~~~
pytest ml_project/tests
~~~

## Структура проекта

.
├── ml_project                                             <- Source code for use in this project                                  
│   ├── configs                                            <- Configs for train and predict
│   │   ├── config_logistic_regr.yaml
│   │   ├── config_predict.yaml
│   │   └── config_random_forest.yaml
│   ├── data                                               <- Data dir
│   │   ├── data.csv
│   │   ├── test.csv
│   │   ├── test_prediction.txt
│   │   └── test_target.csv
│   ├── entities                                           <- Entities
│   │   ├── feature_params.py
│   │   ├── __init__.py
│   │   ├── predict_pipeline_params.py
│   │   ├── split_params.py
│   │   ├── train_params.py
│   │   └── train_pipeline_params.py                        
│   ├── __init__.py
│   ├── models                                              <- Models
│   │   ├── __init__.py
│   │   ├── model_fit_predict.py                            <- Functions for fit, predict model
│   │   ├── model.pkl
│   ├── notebooks                                           <- Jupyter notebooks.
│   │   └── Train_model.ipynb
│   ├── predict_pipeline.py                                 <- Scripts to predict pipeline
│   ├── tests                                               <- Run tests
│   │   ├── config                                          <- Help configs for run tests
│   │   │   ├── config_for_test_predict.yaml
│   │   │   └── config_for_test.yaml
│   │   ├── for_tests                                       <- Help dir (git ignore)
│   │   ├── generate_dataset.py                             <- Script for generate dataset
│   │   ├── __init__.py
│   │   ├── test_for_predict.py                             <- Tests for predict
│   │   └── test_for_train.py                               <- Tests for train
│   ├── train_pipeline.py                                   <- Scripts to train pipeline
│   └── utils                                               <- Utils functions
│       ├── __init__.py
│       ├── make_dataset.py                                 <- Functions for make dataset
│       ├── preprocessing.py                                <- Functions for preprocessing features
├── README.md                                               <- The top-level README for developers using this project.
├── requirements.txt                                        <- The requirements file for reproducing the analysis environment, e.g.
│                                                              generated with `pip freeze > requirements.txt`
└── setup.py                                                <- makes project pip installable (pip install -e .) so src can be imported
