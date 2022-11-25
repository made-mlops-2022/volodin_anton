MLOps homework2
==============================

Сборка Docker:
```bash
docker build -t hw2_image .
```
Запуск Docker:
```bash
docker run -d --name hw2_container -p 8000:8000 hw2_image
```
Загрузка из https://hub.docker.com/
```bash
docker pull trew1237/hw2_image:latest
```
Запуск:
```bash
docker run -p 8000:8000 trew1237/hw2_image:latest
```
Запуск запросов:
```bash
make_request.py
```
Запуск тестов:
```bash
 python -m pytest test.py 
 ```
