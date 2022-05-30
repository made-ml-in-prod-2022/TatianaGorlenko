Команды запускаются из корневой директории проекта

Собрать докер-образ:
~~~
docker build -f online_inference/Dockerfile -t tgorlenko/online_inference:v1 .
~~~

Загрузить докер-образ:
~~~
docker pull tgorlenko/online_inference:v1
~~~

Контейнер:
~~~
docker run -p 8000:8000 tgorlenko/online_inference:v1
~~~

Пуляем запросы:
~~~
python online_inference/make_request.py
~~~

