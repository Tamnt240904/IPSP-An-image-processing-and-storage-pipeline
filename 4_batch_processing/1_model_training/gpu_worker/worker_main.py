"""
Worker Main Entry Point - GPU worker that consumes and processes training jobs
This script runs on local laptop/GPU machine and processes training jobs from Kafka
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from job_worker import JobWorker

# Ensure output is flushed immediately for better log visibility
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    # Python < 3.7 doesn't have reconfigure, use flush instead
    pass


def main():
    """Main worker loop: consume and process training jobs from Kafka"""
    print("=" * 80)
    print("YOLO Training Job Worker - GPU Worker")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Kafka Broker: {Config.KAFKA_BROKER}")
    print(f"  Training Job Topic: {Config.TRAINING_JOB_TOPIC}")
    print("=" * 80)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"\n✓ GPU detected: {device_name} (count: {device_count})")
        else:
            print("\n⚠️  WARNING: No GPU detected. Training will run on CPU (very slow).")
    except ImportError:
        print("\n⚠️  WARNING: PyTorch not available. GPU check skipped.")
    except Exception as e:
        print(f"\n⚠️  WARNING: Error checking GPU: {e}")
    
    # Initialize job worker
    print("\nInitializing job worker...")
    try:
        # Use unique consumer group ID to process all jobs from beginning
        import time as time_module
        consumer_group = os.environ.get("KAFKA_CONSUMER_GROUP", f"yolo-worker-{int(time_module.time())}")
        
        worker = JobWorker(
            kafka_broker=Config.KAFKA_BROKER,
            training_job_topic=Config.TRAINING_JOB_TOPIC,
            consumer_group=consumer_group
        )
        print("✓ Job worker initialized successfully")
        print(f"  Consumer group: {consumer_group}")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize job worker: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Worker loop
    job_count = 0
    poll_timeout = 60  # seconds
    
    try:
        print("\n" + "=" * 80)
        print("Worker ready. Waiting for training jobs...")
        print("=" * 80)
        print(f"Poll timeout: {poll_timeout} seconds")
        print("Press Ctrl+C to stop the worker\n")
        
        while True:
            try:
                # Consume a training job
                job_spec = worker.consume_job(timeout_seconds=poll_timeout)
                
                if job_spec:
                    job_count += 1
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing job #{job_count}")
                    
                    # Process the job
                    success = worker.process_job(job_spec)
                    
                    if success:
                        print(f"\n✓ Job #{job_count} completed successfully!")
                    else:
                        print(f"\n✗ Job #{job_count} failed!")
                    
                    print("\n" + "=" * 80)
                    print("Waiting for next training job...")
                    print("=" * 80 + "\n")
                else:
                    # No job received (timeout)
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No jobs available. Continuing to poll...")
                
            except KeyboardInterrupt:
                print("\n\nShutting down gracefully...")
                break
            except Exception as e:
                print(f"\n✗ ERROR processing job: {e}")
                import traceback
                traceback.print_exc()
                print("\nWaiting 10 seconds before retrying...")
                time.sleep(10)
                
    finally:
        # Clean up
        worker.close()
        print(f"\nWorker shutdown complete. Processed {job_count} job(s).")


if __name__ == "__main__":
    try:
        print("=" * 80)
        print("Starting YOLO Training Job Worker...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print("=" * 80)
        
        # Test critical imports
        print("\nTesting imports...")
        try:
            import torch
            print(f"✓ PyTorch version: {torch.__version__}")
        except ImportError as e:
            print(f"✗ PyTorch import failed: {e}")
            print("PyTorch is required for GPU training.")
            sys.exit(1)
        
        try:
            from ultralytics import YOLO
            print("✓ Ultralytics imported")
        except ImportError as e:
            print(f"✗ Ultralytics import failed: {e}")
            sys.exit(1)
        
        try:
            from minio import Minio
            print("✓ MinIO imported")
        except ImportError as e:
            print(f"✗ MinIO import failed: {e}")
            sys.exit(1)
        
        try:
            from confluent_kafka import Consumer
            print("✓ confluent-kafka imported")
        except ImportError as e:
            print(f"✗ confluent-kafka import failed: {e}")
            print("Install with: pip install confluent-kafka")
            sys.exit(1)
        
        try:
            from pyspark.sql import SparkSession
            print("✓ PySpark imported")
        except ImportError as e:
            print(f"✗ PySpark import failed: {e}")
            print("Install with: pip install pyspark")
            sys.exit(1)
        
        print("✓ All imports successful\n")
        
        # Run main
        main()
        
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
