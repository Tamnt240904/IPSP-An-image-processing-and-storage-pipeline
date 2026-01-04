@echo off
REM Script to run the GPU worker on Windows (local)

REM Set environment variables
REM Use port 19092 for external access (Kafka external listener)
set KAFKA_BROKER=localhost:19092
set TRAINING_JOB_TOPIC=training_jobs
set MINIO_ENDPOINT=localhost:9000
set MINIO_ACCESS_KEY=bigdataproject
set MINIO_SECRET_KEY=bigdataproject
set MINIO_BUCKET=traffic-models

REM Optional: Set Spark memory (default = 6g)
REM set SPARK_DRIVER_MEMORY=6g

REM These flags allow Spark to access internal Java APIs required for memory management
set JVM_OPTS=--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED

REM Apply JVM options to PySpark (must be set before Python imports Spark)
set PYSPARK_SUBMIT_ARGS=--driver-java-options "%JVM_OPTS%" --conf "spark.executor.extraJavaOptions=%JVM_OPTS%" pyspark-shell

REM Run the worker
python worker_main.py

