"""
Data processing module for converting Kafka data to YOLO training format
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, FloatType


class YOLODataProcessor:
    """Process Kafka data and convert to YOLO format"""
    
    def __init__(self, spark: SparkSession, data_dir: str):
        """
        Initialize data processor
        
        Args:
            spark: SparkSession instance
            data_dir: Directory to store processed data
        """
        self.spark = spark
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLO dataset structure with train/test split
        self.images_train_dir = self.data_dir / "images" / "train"
        self.images_val_dir = self.data_dir / "images" / "val"
        self.labels_train_dir = self.data_dir / "labels" / "train"
        self.labels_val_dir = self.data_dir / "labels" / "val"
        self.images_train_dir.mkdir(parents=True, exist_ok=True)
        self.images_val_dir.mkdir(parents=True, exist_ok=True)
        self.labels_train_dir.mkdir(parents=True, exist_ok=True)
        self.labels_val_dir.mkdir(parents=True, exist_ok=True)
        
    def get_kafka_schema(self) -> StructType:
        """Define schema for Kafka messages"""
        object_schema = StructType([
            StructField("class_id", IntegerType()),
            StructField("bbox", ArrayType(FloatType()))
        ])
        
        kafka_schema = StructType([
            StructField("camera_id", StringType()),
            StructField("image_id", StringType()),
            StructField("image", StringType()),  # base64 encoded
            StructField("objects", ArrayType(object_schema))
        ])
        
        return kafka_schema
    
    def count_new_messages(self, kafka_broker: str, topic: str) -> int:
        """
        Count messages in Kafka
        
        Args:
            kafka_broker: Kafka broker address
            topic: Topic name
            
        Returns:
            Number of messages
        """
        try:
            # Count all messages
            raw_df = self.spark.read \
                .format("kafka") \
                .option("kafka.bootstrap.servers", kafka_broker) \
                .option("subscribe", topic) \
                .option("startingOffsets", "earliest") \
                .option("endingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .load()
            
            count = raw_df.count()
            return count
        except Exception as e:
            print(f"Error counting messages: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def read_from_kafka(self, kafka_broker: str, topic: str) -> DataFrame:
        """
        Read batch data from Kafka (all available data)
        
        Args:
            kafka_broker: Kafka broker address
            topic: Topic name
            
        Returns:
            DataFrame with parsed Kafka messages
        """
        # Read all available data
        raw_df = self.spark.read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .option("endingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        schema = self.get_kafka_schema()
        parsed_df = raw_df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    
    def process_dataframe(self, df: DataFrame) -> int:
        """
        Process entire DataFrame and convert to YOLO format
        Splits data: first 100 images per camera -> test set, rest -> train set
        
        Args:
            df: Spark DataFrame with Kafka data
            
        Returns:
            Number of successfully processed images
        """
        total_count = df.count()
        
        if total_count == 0:
            print("No rows to process")
            return 0
        
        print(f"Processing {total_count} images...")
        print("Splitting data: first 100 images per camera -> test set, rest -> train set")
        
        # Number of images per camera
        window_spec = Window.partitionBy("camera_id").orderBy("image_id")
        df_with_rownum = df.withColumn("row_num", row_number().over(window_spec))
        
        # Cache the dataframe after window operation to avoid recomputation
        df_with_rownum.cache()
        
        # Split into test (first 100 images per camera) and train (rest)
        test_df = df_with_rownum.filter(col("row_num") <= 100)
        train_df = df_with_rownum.filter(col("row_num") > 100)
        
        print("Data split defined. Processing test set first (first 100 images per camera)...")
        print("Train set will be processed next (images from 101st onwards per camera)...")
        
        # Use accumulator to track processed count across partitions
        from pyspark import AccumulatorParam
        
        class IntAccumulatorParam(AccumulatorParam):
            def zero(self, initialValue):
                return 0
            def addInPlace(self, v1, v2):
                return v1 + v2
        
        processed_count_acc = self.spark.sparkContext.accumulator(0, IntAccumulatorParam())
        
        # Broadcast directory paths to ensure all partitions can access them
        images_train_dir_bc = self.spark.sparkContext.broadcast(str(self.images_train_dir))
        images_val_dir_bc = self.spark.sparkContext.broadcast(str(self.images_val_dir))
        labels_train_dir_bc = self.spark.sparkContext.broadcast(str(self.labels_train_dir))
        labels_val_dir_bc = self.spark.sparkContext.broadcast(str(self.labels_val_dir))
        
        def process_partition(partition, images_dir_bc, labels_dir_bc, set_name: str):
            """Process a partition of rows (unified for train/test)"""
            import base64 as b64
            import io
            import os
            from pathlib import Path as PPath
            from PIL import Image
            
            images_dir = images_dir_bc.value
            labels_dir = labels_dir_bc.value
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            partition_count = 0
            partition_total = 0
            
            for row in partition:
                partition_total += 1
                try:
                    image_id = row.image_id if hasattr(row, 'image_id') else str(row['image_id'])
                    image_base64 = row.image if hasattr(row, 'image') else row['image']
                    objects = row.objects if hasattr(row, 'objects') and row.objects else (row.get('objects', []) if isinstance(row, dict) else [])
                    
                    if not image_base64:
                        continue
                    
                    # Decode and save image
                    try:
                        image_bytes = b64.b64decode(image_base64)
                        image = Image.open(io.BytesIO(image_bytes))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image_path = PPath(images_dir) / f"{image_id}.jpg"
                        image.save(image_path, "JPEG")
                    except Exception as e:
                        print(f"Error decoding image {image_id}: {e}")
                        continue
                    
                    # Convert objects to list of dicts
                    objects_list = []
                    if objects:
                        for obj in objects:
                            if hasattr(obj, 'asDict'):
                                obj_dict = obj.asDict()
                            elif hasattr(obj, '__dict__'):
                                obj_dict = obj.__dict__
                            elif isinstance(obj, dict):
                                obj_dict = obj
                            else:
                                continue
                            objects_list.append({
                                'class_id': int(obj_dict.get('class_id', 0)),
                                'bbox': obj_dict.get('bbox', [])
                            })
                    
                    # Create label file
                    try:
                        label_path = PPath(labels_dir) / f"{image_id}.txt"
                        with open(label_path, 'w') as f:
                            for obj in objects_list:
                                try:
                                    class_id = int(obj.get('class_id', 0))
                                    bbox = obj.get('bbox', [])
                                    if bbox and len(bbox) == 4:
                                        bbox = [float(x) for x in bbox]
                                        bbox = [max(0.0, min(1.0, x)) for x in bbox]
                                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                                except (ValueError, TypeError, IndexError) as e:
                                    print(f"Error processing object in {image_id}: {e}")
                                    continue
                    except Exception as e:
                        print(f"Error creating label file for {image_id}: {e}")
                        continue
                    
                    partition_count += 1
                except Exception as e:
                    print(f"Error processing row in partition: {e}")
                    continue
                
                if partition_total % 50 == 0:
                    print(f"Partition progress ({set_name}): {partition_total} rows processed, {partition_count} successful")
            
            processed_count_acc.add(partition_count)
        
        # Process test set (first 100 images per camera)
        print(f"\nProcessing test set (first 100 images per camera)...")
        test_df.select("camera_id", "image_id", "image", "objects").foreachPartition(
            lambda p: process_partition(p, images_val_dir_bc, labels_val_dir_bc, "test")
        )
        test_count_processed = processed_count_acc.value
        
        # Process train set (images after 100th per camera)
        print(f"\nProcessing train set (images from 101st onwards per camera)...")
        train_df.select("camera_id", "image_id", "image", "objects").foreachPartition(
            lambda p: process_partition(p, images_train_dir_bc, labels_train_dir_bc, "train")
        )
        
        total_processed = processed_count_acc.value
        train_count_processed = total_processed - test_count_processed
        
        print(f"\nSuccessfully processed:")
        print(f"  Test set: {test_count_processed} images")
        print(f"  Train set: {train_count_processed} images")
        print(f"  Total: {total_processed}/{total_count} images")
        
        # Unpersist cached dataframe to free memory
        df_with_rownum.unpersist()
        
        return total_processed
    
    def create_dataset_yaml(self, dataset_name: str = "traffic_dataset") -> str:
        """
        Create YOLO dataset configuration YAML file
        Uses train/test split: first 100 images per camera -> test set, rest -> train set
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to created YAML file
        """
        dataset_yaml = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',  # First 100 images per camera
            'names': {
                0: 'motorcycle',
                1: 'car',
                2: 'bus',
                3: 'truck'
            },
            'nc': 4  # Number of classes
        }
        
        yaml_path = self.data_dir / f"{dataset_name}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        return str(yaml_path)

