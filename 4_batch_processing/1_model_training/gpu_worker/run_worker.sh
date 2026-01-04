#!/bin/bash
# Simple script to run the GPU worker
# This script runs on your local laptop/GPU machine

# Set environment variables
export KAFKA_BROKER="localhost:19092"         # Kafka external listener via port forwarding
export MINIO_ENDPOINT="localhost:9000"

# Java 17+ compatibility - Set JVM options BEFORE Spark starts
JVM_OPTS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
export PYSPARK_SUBMIT_ARGS="--driver-java-options \"${JVM_OPTS}\" --conf \"spark.executor.extraJavaOptions=${JVM_OPTS}\" pyspark-shell"
export TRAINING_JOB_TOPIC="training_jobs"      # Topic for training jobs
export MINIO_ENDPOINT="localhost:9000"         # MinIO via port forwarding
export MINIO_ACCESS_KEY="bigdataproject"       # MinIO access key (same as MINIO_ROOT_USER)
export MINIO_SECRET_KEY="bigdataproject"       # MinIO secret key (same as MINIO_ROOT_PASSWORD)
export MINIO_BUCKET="traffic-models"           # MinIO bucket name

# Optional: Set Spark memory (default = 6g)
# export SPARK_DRIVER_MEMORY="6g"

# Run the worker
python3 worker_main.py

