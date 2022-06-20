Чтобы развернуть airflow, предварительно собрав контейнеры, необходимо ввести команды

~~~
# путь к директории, куда будут сохраняться данные:
export HOST_DATA_DIR=/tmp/data
# для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
~~~
