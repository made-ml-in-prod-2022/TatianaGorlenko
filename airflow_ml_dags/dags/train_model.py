import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.models import Variable


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
host_data_dir = Variable.get("host_data_dir")

with DAG(
        "train_model_docker",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:

    prepare_data = DockerOperator(
        image="airflow-prepare-data",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-prepare-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=host_data_dir, target="/data", type='bind')]
    )

    split_data = DockerOperator(
        image="airflow-split-data",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/splitted/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=host_data_dir, target="/data", type='bind')]
    )   
    
    train_model = DockerOperator(
        image="airflow-train-model",
        command="--input-dir /data/splitted/{{ ds }} --output-dir /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=host_data_dir, target="/data", type='bind')]
    )

    validate_model = DockerOperator(
        image="airflow-validate-model",
        command="--input-dir /data/splitted/{{ ds }} --models-dir /data/models/{{ ds }} --output-dir /data/metrics/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-validate-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=host_data_dir, target="/data", type='bind')]
    )   

prepare_data >> split_data >> train_model >> validate_model
