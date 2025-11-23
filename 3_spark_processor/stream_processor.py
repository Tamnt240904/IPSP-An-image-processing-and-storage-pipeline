import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
# ĐÃ SỬA: Bỏ ObjectType khỏi dòng này
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "traffic-metadata"

# 2. Định nghĩa Schema
json_schema = StructType([
    StructField("record_key", StringType(), True),
    StructField("camera_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("lmdb_info", StructType([
        StructField("lmdb_filepath", StringType(), True),
        StructField("frame_height", IntegerType(), True),
        StructField("frame_width", IntegerType(), True)
    ])),
    StructField("schema_version", StringType(), True)
])

def create_spark_session():
    return SparkSession.builder \
        .appName("TrafficAnalysis_V1_Debug") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1") \
        .master("local[*]") \
        .getOrCreate()

def run_spark_job():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("🚀 Đang khởi động Spark Streaming...")
    print(f"📡 Kết nối Kafka: {KAFKA_BOOTSTRAP_SERVERS}")

    # --- 1. ĐỌC DỮ LIỆU ---
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", TOPIC_NAME) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 50) \
        .load()

    # --- 2. CHUYỂN ĐỔI ---
    processed_df = kafka_df.select(
        from_json(col("value").cast("string"), json_schema).alias("data")
    ).select("data.*")

    # --- 3. GHI KẾT QUẢ ---
    query = processed_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()

    print("✅ Đang chạy... (Chờ một chút để Spark tải thư viện và hiện bảng)")
    query.awaitTermination()

if __name__ == "__main__":
    run_spark_job()