"""
Job Dispatcher Module - Creates and publishes training jobs to Kafka
Modular component for dispatching training jobs without GPU dependency
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from pyspark.sql import SparkSession
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data_processor import YOLODataProcessor


class JobDispatcher:
    """Dispatches training jobs to Kafka for GPU workers to process"""
    
    def __init__(self, kafka_broker: str, training_job_topic: str):
        """
        Initialize job dispatcher
        
        Args:
            kafka_broker: Kafka broker address
            training_job_topic: Topic name for publishing training jobs
        """
        self.kafka_broker = kafka_broker
        self.training_job_topic = training_job_topic
        self.producer = None
        self._init_producer()
    
    def _init_producer(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_broker,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True
            )
            print(f"✓ Kafka producer initialized for topic: {self.training_job_topic}")
        except Exception as e:
            print(f"✗ Failed to initialize Kafka producer: {e}")
            raise
    
    def check_new_data_threshold(self, spark: SparkSession) -> Tuple[bool, int]:
        """
        Check if there are enough new images to trigger training
        
        Args:
            spark: SparkSession instance
            
        Returns:
            Tuple of (should_train: bool, new_count: int)
        """
        print("\n" + "=" * 80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for new data...")
        print("=" * 80)
        
        data_processor = YOLODataProcessor(spark, Config.DATA_DIR)
        
        try:
            # Count all messages (simplified - can be enhanced with state tracking)
            new_count = data_processor.count_new_messages(
                Config.KAFKA_BROKER,
                Config.TOPIC_NAME
            )
            
            print(f"Total images available: {new_count}")
            print(f"Threshold: {Config.NEW_DATA_THRESHOLD}")
            
            should_train = new_count >= Config.NEW_DATA_THRESHOLD
            
            if should_train:
                print(f"✓ Threshold met! ({new_count} >= {Config.NEW_DATA_THRESHOLD})")
                print("Creating training job...")
            else:
                print(f"✗ Threshold not met ({new_count} < {Config.NEW_DATA_THRESHOLD})")
                print("Skipping training job creation.")
            
            return should_train, new_count
            
        except Exception as e:
            print(f"Error checking new data: {e}")
            import traceback
            traceback.print_exc()
            return False, 0
    
    def create_training_job(self, record_count: int) -> Dict[str, Any]:
        """
        Create a training job specification
        
        Args:
            record_count: Number of records available for training
            
        Returns:
            Dictionary with training job specification
        """
        job_id = f"training_job_{int(time.time() * 1000)}"
        
        job_spec = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "data_source": {
                "kafka_broker": Config.KAFKA_BROKER,
                "topic": Config.TOPIC_NAME,
                "record_count": record_count
            },
            "training_config": {
                "epochs": Config.EPOCHS,
                "batch_size": Config.BATCH_SIZE,
                "img_size": Config.IMG_SIZE,
                "model_size": Config.MODEL_SIZE
            },
            "storage_config": {
                "minio_endpoint": Config.MINIO_ENDPOINT,
                "minio_access_key": Config.MINIO_ACCESS_KEY,
                "minio_secret_key": Config.MINIO_SECRET_KEY,
                "minio_bucket": Config.MINIO_BUCKET,
                "minio_secure": Config.MINIO_SECURE
            },
            "paths": {
                "data_dir": Config.DATA_DIR,
                "output_dir": Config.OUTPUT_DIR
            }
        }
        
        return job_spec
    
    def publish_training_job(self, job_spec: Dict[str, Any]) -> bool:
        """
        Publish training job to Kafka topic
        
        Args:
            job_spec: Training job specification dictionary
            
        Returns:
            True if published successfully, False otherwise
        """
        try:
            job_id = job_spec["job_id"]
            
            print(f"\nPublishing training job to topic '{self.training_job_topic}'...")
            print(f"Job ID: {job_id}")
            
            # Publish to Kafka
            future = self.producer.send(
                self.training_job_topic,
                key=job_id,
                value=job_spec
            )
            
            # Wait for the message to be sent
            record_metadata = future.get(timeout=10)
            
            print(f"✓ Training job published successfully!")
            print(f"  Topic: {record_metadata.topic}")
            print(f"  Partition: {record_metadata.partition}")
            print(f"  Offset: {record_metadata.offset}")
            
            return True
            
        except KafkaError as e:
            print(f"✗ Kafka error publishing training job: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"✗ Error publishing training job: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def dispatch_job(self, spark: SparkSession) -> bool:
        """
        Main dispatch logic: check data, create job, publish to Kafka
        
        Args:
            spark: SparkSession instance
            
        Returns:
            True if job was dispatched successfully, False otherwise
        """
        try:
            # Check if we should create a training job
            should_train, record_count = self.check_new_data_threshold(spark)
            
            if not should_train:
                return False
            
            # Create training job specification
            job_spec = self.create_training_job(record_count)
            
            # Publish to Kafka
            success = self.publish_training_job(job_spec)
            
            return success
            
        except Exception as e:
            print(f"✗ Error in dispatch_job: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
                print("✓ Kafka producer closed")
            except Exception as e:
                print(f"Warning: Error closing producer: {e}")
