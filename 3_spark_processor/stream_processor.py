import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, window, avg, to_timestamp, get_json_object, explode, sum as _sum
from pyspark.sql.types import *
from udf_logic import process_traffic_analysis

MONGO_URI = os.environ.get("MONGO_URI")
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
TOPIC_NAME = os.environ.get("TOPIC_NAME", "traffic_data")

spark = SparkSession.builder \
    .appName("TrafficProcessor_v17") \
    .config("spark.mongodb.write.connection.uri", MONGO_URI) \
    .config("spark.mongodb.write.database", "traffic_db") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

object_schema = StructType([
    StructField("track_id", IntegerType()),
    StructField("class_id", IntegerType()),
    StructField("world_coordinates", StructType([
        StructField("x", FloatType()),
        StructField("y", FloatType())
    ])),
    StructField("segmentation", ArrayType(ArrayType(ArrayType(IntegerType()))))
])

kafka_schema = StructType([
    StructField("camera_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("objects", ArrayType(object_schema))
])

process_udf = udf(process_traffic_analysis, StringType())

raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "earliest") \
    .load()

parsed_df = raw_df.select(
    from_json(col("value").cast("string"), kafka_schema).alias("data")
).select("data.*")

enriched_df = parsed_df.withColumn(
    "insights_json",
    process_udf(col("camera_id"), col("timestamp"), col("objects"))
)

stream_with_ts = enriched_df.withColumn("event_time", to_timestamp(col("timestamp"))) \
    .withWatermark("event_time", "30 seconds")

processed_stats_df = stream_with_ts \
    .withColumn("density", get_json_object(col("insights_json"), "$.density.percentage").cast("float")) \
    .withColumn("m_unique", get_json_object(col("insights_json"), "$.unique_counts.motorcycle").cast("int")) \
    .withColumn("c_unique", get_json_object(col("insights_json"), "$.unique_counts.car").cast("int")) \
    .withColumn("b_unique", get_json_object(col("insights_json"), "$.unique_counts.bus").cast("int")) \
    .withColumn("t_unique", get_json_object(col("insights_json"), "$.unique_counts.truck").cast("int")) \
    .withColumn("m_speed", get_json_object(col("insights_json"), "$.avg_speeds_frame.avg_speed_motorcycle").cast("float")) \
    .withColumn("c_speed", get_json_object(col("insights_json"), "$.avg_speeds_frame.avg_speed_car").cast("float")) \
    .withColumn("b_speed", get_json_object(col("insights_json"), "$.avg_speeds_frame.avg_speed_bus").cast("float")) \
    .withColumn("t_speed", get_json_object(col("insights_json"), "$.avg_speeds_frame.avg_speed_truck").cast("float"))

stats_df = processed_stats_df \
    .groupBy(window(col("event_time"), "1 minute", "30 seconds"), col("camera_id")) \
    .agg(
        avg("density").alias("avg_density_pct"),
        _sum("m_unique").alias("unique_motorcycle_count"),
        _sum("c_unique").alias("unique_car_count"),
        _sum("b_unique").alias("unique_bus_count"),
        _sum("t_unique").alias("unique_truck_count"),
        avg("m_speed").alias("avg_speed_motorcycle_kmh"),
        avg("c_speed").alias("avg_speed_car_kmh"),
        avg("b_speed").alias("avg_speed_bus_kmh"),
        avg("t_speed").alias("avg_speed_truck_kmh")
    ) \
    .select(
        col("window.start").alias("start_time"),
        col("camera_id"),
        "avg_density_pct",
        "unique_motorcycle_count", "unique_car_count", "unique_bus_count", "unique_truck_count",
        "avg_speed_motorcycle_kmh", "avg_speed_car_kmh", "avg_speed_bus_kmh", "avg_speed_truck_kmh"
    )

query_per_second = enriched_df.select(col("camera_id"), col("timestamp"), col("insights_json").alias("insights")) \
    .writeStream.format("mongodb").outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint_per_second_v17") \
    .option("spark.mongodb.write.collection", "traffic_per_second").start()

query_per_minute = stats_df \
    .writeStream.format("mongodb").outputMode("complete") \
    .option("checkpointLocation", "/tmp/checkpoint_per_minute_v17") \
    .option("spark.mongodb.write.collection", "traffic_per_minute").start()

spark.streams.awaitAnyTermination()