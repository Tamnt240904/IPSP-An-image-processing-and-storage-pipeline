"""
Job Worker Module - Consumes training jobs from Kafka and executes GPU training
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from confluent_kafka import Consumer, KafkaError, KafkaException
from pyspark.sql import SparkSession

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data_processor import YOLODataProcessor
from model_trainer import YOLOModelTrainer
from minio_uploader import MinIOUploader
from util import create_spark_session, get_checkpoint_path, prepare_upload_files


class JobWorker:
    """Consumes training jobs from Kafka and executes GPU training"""
    
    def __init__(self, kafka_broker: str, training_job_topic: str, consumer_group: str = "yolo-training-workers"):
        """
        Initialize job worker
        
        Args:
            kafka_broker: Kafka broker address
            training_job_topic: Topic name for consuming training jobs
            consumer_group: Kafka consumer group ID
        """
        self.kafka_broker = kafka_broker
        self.training_job_topic = training_job_topic
        self.consumer_group = consumer_group
        self.consumer = None
        self._init_consumer()
    
    def _init_consumer(self):
        """Initialize Kafka consumer using confluent-kafka (production-grade client)"""
        try:
            print(f"Connecting to Kafka broker: {self.kafka_broker}")
            print(f"Subscribing to topic: {self.training_job_topic}")
            
            consumer_config = {
                'bootstrap.servers': self.kafka_broker,
                'group.id': self.consumer_group,
                'auto.offset.reset': 'earliest',  # Start from beginning if no committed offset
                'enable.auto.commit': True,  # Automatically commit offsets after processing
                'session.timeout.ms': 30000,  # Session timeout
                'max.poll.interval.ms': 300000,  # Max time between polls
            }
            
            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe([self.training_job_topic])
            
            print(f"✓ Kafka consumer created (confluent-kafka)")
            print(f"  Consumer group: {self.consumer_group}")
            print(f"  Auto offset reset: earliest")
            print(f"  Auto commit: enabled")
            
            # Poll to trigger assignment and verify connection
            print(f"  Verifying partition assignment...")
            try:
                # Poll with short timeout to trigger assignment
                msg = self.consumer.poll(timeout=2.0)
                
                # Get assigned partitions
                assigned_partitions = self.consumer.assignment()
                if assigned_partitions:
                    print(f"✓ Partitions assigned: {[f'{tp.topic}:{tp.partition}' for tp in assigned_partitions]}")
                    # Log partition info
                    for tp in assigned_partitions:
                        try:
                            # Get committed offset for this partition
                            low, high = self.consumer.get_watermark_offsets(tp, timeout=1.0)
                            print(f"    Partition {tp.partition}: offset range [{low}, {high}]")
                        except Exception as e:
                            print(f"    Partition {tp.partition}: could not get offset info ({e})")
                else:
                    # Wait for assignment 
                    time.sleep(1)
                    assigned_partitions = self.consumer.assignment()
                    if assigned_partitions:
                        print(f"✓ Partitions assigned: {[f'{tp.topic}:{tp.partition}' for tp in assigned_partitions]}")
                    else:
                        print(f"  ⚠️  No partitions assigned yet (this is unusual with confluent-kafka)")
                        print(f"     Topic '{self.training_job_topic}' may not exist or broker issue")
                
                # If we got a message during the test poll, it's fine - we'll process it later
                if msg and not msg.error():
                    print(f"  Note: Found message during initialization (will process in consume cycle)")
                
            except KafkaException as kafka_error:
                print(f"  ✗ Kafka error during initialization: {kafka_error}")
                raise
            except Exception as e:
                print(f"  ✗ Error during initialization: {e}")
                raise
                
        except Exception as e:
            print(f"✗ Failed to initialize Kafka consumer: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def consume_job(self, timeout_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Consume a single training job from Kafka
        
        Args:
            timeout_seconds: Maximum time to wait for a job (in seconds)
            
        Returns:
            Training job specification dictionary, or None if timeout
        """
        try:
            print(f"\nWaiting for training job (timeout: {timeout_seconds}s)...")
            
            # Check partition assignment
            assigned_partitions = self.consumer.assignment()
            if assigned_partitions:
                print(f"  Assigned partitions: {[f'{tp.topic}:{tp.partition}' for tp in assigned_partitions]}")
            else:
                print(f"  ⚠️  No partitions assigned yet (should be assigned with confluent-kafka)")
            
            # Poll for messages with timeout
            start_time = time.time()
            poll_count = 0
            poll_timeout = 1.0  # seconds per poll
            
            while time.time() - start_time < timeout_seconds:
                poll_count += 1
                try:
                    msg = self.consumer.poll(timeout=poll_timeout)
                    
                    if msg is None:
                        continue
                    
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        else:
                            print(f"  ✗ Kafka error: {msg.error()}")
                            continue
                    
                    # Successfully received a message
                    try:
                        # Deserialize the message value (JSON)
                        job_spec = json.loads(msg.value().decode('utf-8'))
                        job_id = msg.key().decode('utf-8') if msg.key() else job_spec.get('job_id', 'unknown')
                        
                        print(f"✓ Received training job: {job_id}")
                        print(f"  Topic: {msg.topic()}")
                        print(f"  Partition: {msg.partition()}")
                        print(f"  Offset: {msg.offset()}")
                        
                        return job_spec
                        
                    except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
                        print(f"  ✗ Error decoding message: {decode_error}")
                        continue
                    
                except KafkaException as kafka_error:
                    print(f"  ✗ Kafka exception during poll: {kafka_error}")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  ✗ Unexpected error during poll: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.5)
            
            print(f"⏱ No job received within {timeout_seconds} seconds (after {poll_count} polls)")
            return None
            
        except KafkaException as e:
            print(f"✗ Kafka exception consuming job: {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"✗ Error consuming job: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_job(self, job_spec: Dict[str, Any]) -> bool:
        """
        Process a training job - execute the complete training pipeline
        
        Args:
            job_spec: Training job specification dictionary
            
        Returns:
            True if job processed successfully, False otherwise
        """
        job_id = job_spec.get('job_id', 'unknown')
        
        print("\n" + "=" * 80)
        print(f"Processing Training Job: {job_id}")
        print("=" * 80)
        print(f"Job created at: {job_spec.get('created_at', 'N/A')}")
        print()
        
        try:
            # Extract configuration from job spec
            data_config = job_spec.get('data_source', {})
            training_config = job_spec.get('training_config', {})
            storage_config = job_spec.get('storage_config', {})
            paths_config = job_spec.get('paths', {})
            
            kafka_broker = os.environ.get('KAFKA_BROKER') or data_config.get('kafka_broker', Config.KAFKA_BROKER)
            topic_name = data_config.get('topic', Config.TOPIC_NAME)
            data_dir = paths_config.get('data_dir', Config.DATA_DIR)
            output_dir = paths_config.get('output_dir', Config.OUTPUT_DIR)
            
            epochs = training_config.get('epochs', Config.EPOCHS)
            batch_size = training_config.get('batch_size', Config.BATCH_SIZE)
            img_size = training_config.get('img_size', Config.IMG_SIZE)
            model_size = training_config.get('model_size', Config.MODEL_SIZE)
            
            # MinIO endpoint
            minio_endpoint = os.environ.get('MINIO_ENDPOINT') or storage_config.get('minio_endpoint', Config.MINIO_ENDPOINT)
            minio_access_key = os.environ.get('MINIO_ACCESS_KEY') or storage_config.get('minio_access_key', Config.MINIO_ACCESS_KEY)
            minio_secret_key = os.environ.get('MINIO_SECRET_KEY') or storage_config.get('minio_secret_key', Config.MINIO_SECRET_KEY)
            minio_bucket = os.environ.get('MINIO_BUCKET') or storage_config.get('minio_bucket', Config.MINIO_BUCKET)
            minio_secure = storage_config.get('minio_secure', Config.MINIO_SECURE)
            
            print(f"Job Configuration:")
            print(f"  Kafka Broker: {kafka_broker} {'(from env)' if os.environ.get('KAFKA_BROKER') else '(from job spec)'}")
            print(f"  Topic: {topic_name}")
            print(f"  Data Dir: {data_dir}")
            print(f"  Output Dir: {output_dir}")
            print(f"  Epochs: {epochs}")
            print(f"  Batch Size: {batch_size}")
            print(f"  Image Size: {img_size}")
            print(f"  Model Size: {model_size}")
            print(f"  MinIO Endpoint: {minio_endpoint} {'(from env)' if os.environ.get('MINIO_ENDPOINT') else '(from job spec)'}")
            print(f"  MinIO Bucket: {minio_bucket}")
            print()
            
            # Initialize directories
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize MinIO uploader
            print("Initializing MinIO connection...")
            uploader = MinIOUploader(
                endpoint=minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                bucket=minio_bucket,
                secure=minio_secure
            )
            print("✓ MinIO connection initialized")
            
            # Create Spark session
            print("Creating Spark session...")
            spark = self._create_spark_session()
            print("✓ Spark session created")
            
            try:
                # Step 1: Read data from Kafka
                print("\n[Step 1] Reading data from Kafka...")
                data_processor = YOLODataProcessor(spark, data_dir)
                kafka_df = data_processor.read_from_kafka(kafka_broker, topic_name)
                
                record_count = kafka_df.count()
                print(f"Retrieved {record_count} total records from Kafka")
                
                if record_count == 0:
                    print("✗ No data found in Kafka. Skipping job.")
                    return False
                
                # Step 2: Convert to YOLO format
                print("\n[Step 2] Converting data to YOLO format...")
                processed_count = data_processor.process_dataframe(kafka_df)
                
                if processed_count == 0:
                    print("✗ No data was successfully processed. Skipping job.")
                    return False
                
                # Step 3: Create dataset YAML
                print("\n[Step 3] Creating dataset configuration...")
                dataset_yaml = data_processor.create_dataset_yaml()
                print(f"Dataset YAML created: {dataset_yaml}")
                
                # Step 4: Check for existing checkpoint
                print("\n[Step 4] Checking for existing checkpoint...")
                resume_from = get_checkpoint_path(uploader, output_dir)
                if resume_from:
                    print(f"Resuming training from checkpoint: {resume_from}")
                else:
                    print("No existing checkpoint found, starting fresh training")
                
                # Step 5: Train YOLO model
                print("\n[Step 5] Training YOLO model...")
                trainer = YOLOModelTrainer(
                    dataset_yaml=dataset_yaml,
                    output_dir=output_dir,
                    model_size=model_size,
                    resume_from=resume_from
                )
                
                training_metadata = trainer.train(
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size
                )
                
                print("\nTraining completed!")
                print(f"Best weights: {training_metadata.get('best_weights')}")
                print(f"Last weights: {training_metadata.get('last_weights')}")
                
                # Step 6: Collect output files
                print("\n[Step 6] Collecting output files...")
                output_files = trainer.get_output_files()
                
                for key, path in output_files.items():
                    if path and path.exists():
                        print(f"  Found: {key} -> {path}")
                
                # Prepare files for upload
                upload_files = prepare_upload_files(output_dir, output_files)
                
                # Step 7: Upload to MinIO
                print("\n[Step 7] Uploading outputs to MinIO...")
                upload_results = uploader.upload_training_outputs(upload_files)
                dataset_upload_success = uploader.upload_dataset_yaml(dataset_yaml)
                
                print("\nUpload results:")
                for file_type, success in upload_results.items():
                    status = "✓" if success else "✗"
                    print(f"  {status} {file_type}")
                print(f"  {'✓' if dataset_upload_success else '✗'} dataset.yaml")
                
                print("\n" + "=" * 80)
                print(f"Training job {job_id} completed successfully!")
                print("=" * 80)
                return True
                
            finally:
                # Stop Spark session
                print("Stopping Spark session...")
                spark.stop()
                print("✓ Spark session stopped")
                
        except Exception as e:
            print(f"\n✗ Error processing job {job_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_spark_session(self) -> SparkSession:
        """Create Spark session for data processing"""
        return create_spark_session(
            app_name="YOLO_Training_Worker",
            master="local[*]"
        )
    
    def close(self):
        """Close Kafka consumer"""
        if self.consumer:
            try:
                # Unsubscribe before closing
                self.consumer.unsubscribe()
                self.consumer.close()
                print("✓ Kafka consumer closed")
            except Exception as e:
                print(f"Warning: Error closing consumer: {e}")

