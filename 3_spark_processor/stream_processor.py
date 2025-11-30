import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType, FloatType

# --- CẤU HÌNH ---
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "traffic-metadata"
# Cấu hình MongoDB
MONGO_URI = "mongodb://root:bigdataproject@localhost:27017/traffic_db.analysis_results?authSource=admin"

# Schema (Giữ nguyên)
kafka_schema = StructType([
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

ai_output_schema = StructType([
    StructField("counts", MapType(StringType(), IntegerType()), True),
    StructField("density", FloatType(), True),
    StructField("status", StringType(), True),
    StructField("traffic_status", StringType(), True),
    StructField("metrics_debug", MapType(StringType(), IntegerType()), True),
    StructField("error", StringType(), True)
])

# Import logic AI (Cần import sau khi addPyFile ở thực tế, nhưng ở local thì import luôn cũng được)
# Tuy nhiên để chắc chắn, ta giữ nguyên logic addPyFile
from udf_logic import process_image_logic

def create_spark_session():
    return SparkSession.builder \
        .appName("TrafficAnalysis_V2_Final_Mongo") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1,org.mongodb.spark:mongo-spark-connector_2.13:10.3.0") \
        .config("spark.sql.shuffle.partitions", "4") \
        .master("local[*]") \
        .getOrCreate()

def run_spark_job():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("🚀 Đang khởi động Spark Streaming (Sink: MongoDB)...")

    # Gửi file UDF cho Workers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    udf_path = os.path.join(current_dir, "udf_logic.py")
    spark.sparkContext.addPyFile(udf_path)

    # Đăng ký UDF
    run_ai_udf = udf(process_image_logic, ai_output_schema)

    # Đọc Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", TOPIC_NAME) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 5) \
        .load()

    # Parse JSON
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), kafka_schema).alias("data")
    ).select("data.*")

    # Chạy AI
    print("⏳ Đang xử lý AI...")
    ai_df = parsed_df.withColumn("ai_result", run_ai_udf(col("record_key")))

    # Flatten kết quả
    final_df = ai_df.select(
        col("record_key").alias("_id"),  # Dùng record_key làm ID chính trong Mongo
        col("camera_id"),
        col("timestamp"),
        col("ai_result.traffic_status").alias("traffic_status"),
        col("ai_result.density").alias("density"),
        col("ai_result.counts").alias("vehicle_counts"),
        col("ai_result.metrics_debug").alias("debug_pixels"),
        col("ai_result.status").alias("proc_status"),
        current_timestamp().alias("processed_at")
    )

    # --- GHI VÀO MONGODB (Thay đổi ở đây) ---
    query = final_df.writeStream \
        .format("mongodb") \
        .option("checkpointLocation", "/tmp/spark_checkpoint_mongo") \
        .option("forceDeleteTempCheckpointLocation", "true") \
        .option("spark.mongodb.connection.uri", MONGO_URI) \
        .option("spark.mongodb.database", "traffic_db") \
        .option("spark.mongodb.collection", "analysis_results") \
        .outputMode("append") \
        .start()

    print("✅ Pipeline đang chạy ngầm và ghi vào MongoDB...")
    query.awaitTermination()

if __name__ == "__main__":
    run_spark_job()