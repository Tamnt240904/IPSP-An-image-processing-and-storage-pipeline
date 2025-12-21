FROM apache/spark:3.4.1
USER root

# Cài đặt thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Nâng cấp pip và cài đặt thư viện
# Thêm opencv-python-headless thay vì opencv-python để giảm dung lượng image
RUN pip3 install --no-cache-dir pymongo minio opencv-python-headless numpy

# Copy source code (Lưu ý: Đảm bảo có cả udf_logic.py trong folder này)
COPY . .

# Chạy Spark với connector và cấu hình v7
CMD ["/opt/spark/bin/spark-submit", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1", \
    "--conf", "spark.sql.execution.pythonUDF.arrow.enabled=false", \
    "stream_processor.py"]