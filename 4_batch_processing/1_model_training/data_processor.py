"""
Data processing module for converting Kafka data to YOLO training format
"""
import os
import json
import base64
import shutil
from pathlib import Path
from typing import List, Dict, Any
import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, udf, explode
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, FloatType
from PIL import Image
import io


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
        
        # YOLO dataset structure
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
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
    
    def read_from_kafka(self, kafka_broker: str, topic: str) -> DataFrame:
        """
        Read batch data from Kafka
        
        Args:
            kafka_broker: Kafka broker address
            topic: Topic name
            
        Returns:
            DataFrame with parsed Kafka messages
        """
        raw_df = self.spark.read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .option("endingOffsets", "latest") \
            .load()
        
        schema = self.get_kafka_schema()
        parsed_df = raw_df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    def decode_image(self, image_base64: str, image_id: str) -> str:
        """
        Decode base64 image and save to disk
        
        Args:
            image_base64: Base64 encoded image string
            image_id: Unique image identifier
            
        Returns:
            Path to saved image file
        """
        try:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image
            image_path = self.images_dir / f"{image_id}.jpg"
            image.save(image_path, "JPEG")
            
            return str(image_path)
        except Exception as e:
            print(f"Error decoding image {image_id}: {e}")
            return None
    
    def create_label_file(self, image_id: str, objects: List[Dict]):
        """
        Create YOLO format label file
        Note: bbox and class_id are already in YOLO format (normalized coordinates)
        
        Args:
            image_id: Unique image identifier
            objects: List of objects with class_id and bbox (already in YOLO format)
        """
        label_path = self.labels_dir / f"{image_id}.txt"
        
        with open(label_path, 'w') as f:
            for obj in objects:
                try:
                    class_id = int(obj.get('class_id', 0))
                    bbox = obj.get('bbox', [])
                    
                    if bbox and len(bbox) == 4:
                        # Ensure bbox values are floats (already normalized in YOLO format)
                        bbox = [float(x) for x in bbox]
                        # Validate values are in [0, 1] range
                        bbox = [max(0.0, min(1.0, x)) for x in bbox]
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Error processing object in {image_id}: {e}")
                    continue
    
    def process_row(self, row) -> bool:
        """
        Process a single row from DataFrame
        
        Args:
            row: Row from Spark DataFrame
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            image_id = row.image_id if hasattr(row, 'image_id') else str(row['image_id'])
            image_base64 = row.image if hasattr(row, 'image') else row['image']
            objects = row.objects if hasattr(row, 'objects') and row.objects else (row.get('objects', []) if isinstance(row, dict) else [])
            
            if not image_base64:
                return False
            
            # Decode and save image
            image_path = self.decode_image(image_base64, image_id)
            if not image_path:
                return False
            
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
                        'bbox': obj_dict.get('bbox', [])  # Already in YOLO format (normalized)
                    })
            
            # Create label file (bbox is already in YOLO format, no conversion needed)
            self.create_label_file(image_id, objects_list)
            
            return True
        except Exception as e:
            print(f"Error processing row: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_dataframe(self, df: DataFrame) -> int:
        """
        Process entire DataFrame and convert to YOLO format
        
        Args:
            df: Spark DataFrame with Kafka data
            
        Returns:
            Number of successfully processed images
        """
        count = 0
        total_count = df.count()
        
        if total_count == 0:
            print("No rows to process")
            return 0
        
        print(f"Processing {total_count} images...")
        
        # Collect all rows and process
        # Note: For very large datasets, consider processing in chunks
        rows = df.collect()
        
        for row in rows:
            if self.process_row(row):
                count += 1
            
            if count % 100 == 0:
                print(f"Processed {count}/{total_count} images...")
        
        print(f"Successfully processed {count}/{total_count} images")
        return count
    
    def create_dataset_yaml(self, dataset_name: str = "traffic_dataset") -> str:
        """
        Create YOLO dataset configuration YAML file
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to created YAML file
        """
        # For training, we'll use all data
        # In production, you might want to split into train/val/test
        dataset_yaml = {
            'path': str(self.data_dir.absolute()),
            'train': 'images',
            'val': 'images',  # Using same as train for simplicity, can be split later
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

