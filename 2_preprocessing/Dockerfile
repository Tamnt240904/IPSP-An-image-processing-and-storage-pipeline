FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install kafka-python minio lmdb msgpack

COPY kafka_producer.py .

CMD ["python", "kafka_producer.py"]