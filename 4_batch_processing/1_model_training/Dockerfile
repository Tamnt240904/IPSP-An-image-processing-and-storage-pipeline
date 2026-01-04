FROM apache/spark:3.4.1
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install typing-extensions compatible with Python 3.8
RUN pip3 install --no-cache-dir typing-extensions==4.5.0


# Install dependencies for dispatcher
RUN pip3 install --no-cache-dir \
    minio \
    Pillow \
    PyYAML \
    kafka-python

# Install PySpark
RUN pip3 install --no-cache-dir pyspark==3.4.1

COPY . .

CMD ["/opt/spark/bin/spark-submit", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1", \
    "--conf", "spark.sql.execution.pythonUDF.arrow.enabled=false", \
    "--driver-memory", "2g", \
    "--executor-memory", "2g", \
    "dispatcher/dispatcher_main.py"]

