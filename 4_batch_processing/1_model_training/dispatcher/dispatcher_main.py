"""
Dispatcher Main Entry Point - Runs job dispatcher
Runs in Kubernetes and dispatches training jobs to Kafka
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from job_dispatcher import JobDispatcher
from util import create_spark_session

# Ensure output is flushed immediately for better log
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass


def main():
    """Main dispatcher loop: check for new data and dispatch training jobs"""
    print("=" * 80)
    print("YOLO Training Job Dispatcher - Continuous Monitoring")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  New Data Threshold: {Config.NEW_DATA_THRESHOLD} images")
    print(f"  Check Interval: {Config.CHECK_INTERVAL_HOURS} hours")
    print(f"  Kafka Broker: {Config.KAFKA_BROKER}")
    print(f"  Input Topic: {Config.TOPIC_NAME}")
    print(f"  Training Job Topic: {Config.TRAINING_JOB_TOPIC}")
    print("=" * 80)
    
    # Initialize job dispatcher
    print("\nInitializing job dispatcher...")
    try:
        dispatcher = JobDispatcher(
            kafka_broker=Config.KAFKA_BROKER,
            training_job_topic=Config.TRAINING_JOB_TOPIC
        )
        print("✓ Job dispatcher initialized successfully")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize job dispatcher: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    check_interval_seconds = Config.CHECK_INTERVAL_HOURS * 3600
    loop_count = 0
    
    # Main loop
    try:
        while True:
            loop_count += 1
            try:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting check cycle #{loop_count}")
                
                print("Creating Spark session...")
                spark = create_spark_session(
                    app_name=Config.SPARK_APP_NAME,
                    master=Config.SPARK_MASTER,
                    driver_memory="2g"
                )
                print("✓ Spark session created")
                
                try:
                    # Dispatch training job if threshold met
                    success = dispatcher.dispatch_job(spark)
                    
                    if success:
                        print(f"\n✓ Training job dispatched successfully!")
                    else:
                        print(f"\nNo training job dispatched (threshold not met or error)")
                    
                finally:
                    print("Stopping Spark session...")
                    spark.stop()
                    print("✓ Spark session stopped")
                
                # Wait for next check
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting {Config.CHECK_INTERVAL_HOURS} hours until next check...")
                print("=" * 80)
                time.sleep(check_interval_seconds)
                
            except KeyboardInterrupt:
                print("\n\nShutting down gracefully...")
                break
            except Exception as e:
                print(f"\n✗ ERROR in dispatcher loop: {e}")
                import traceback
                traceback.print_exc()
                print(f"Waiting 5 minutes before retrying...")
                time.sleep(300)  # Wait 5 minutes on error
                
    finally:
        # Clean up
        dispatcher.close()
        print("\nDispatcher shutdown complete")


if __name__ == "__main__":
    try:
        print("=" * 80)
        print("Starting YOLO Training Job Dispatcher...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print("=" * 80)
        
        main()
        
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


