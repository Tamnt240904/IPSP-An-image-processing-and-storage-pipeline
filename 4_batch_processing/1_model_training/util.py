"""
Utility functions for YOLO training pipeline
"""
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession


def create_spark_session(
    app_name: str = "YOLO_Training",
    master: str = "local[*]",
    driver_memory: Optional[str] = None,
    executor_memory: Optional[str] = None
) -> SparkSession:
    """
    Create and configure Spark session
    
    Args:
        app_name: Spark application name
        master: Spark master URL
        driver_memory: Driver memory (defaults to env var or 6g)
        executor_memory: Executor memory (defaults to driver_memory)
    
    Returns:
        Configured SparkSession
    """
    if driver_memory is None:
        driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    if executor_memory is None:
        executor_memory = driver_memory
    
    # Stop any existing Spark session to ensure new config is applied
    try:
        from pyspark import SparkContext
        sc = SparkContext._active_spark_context
        if sc:
            print("Stopping existing SparkContext to apply new JVM configuration...")
            sc.stop()
    except Exception:
        pass
    
    try:
        existing_spark = SparkSession.getActiveSession()
        if existing_spark:
            print("Stopping existing SparkSession to apply new configuration...")
            existing_spark.stop()
    except Exception:
        pass
    
    ## Set environment variables for JVM options as fallback
    java_opts = [
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    ]
    java_opts_str = " ".join(java_opts)
    
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--driver-java-options \"{java_opts_str}\" --conf \"spark.executor.extraJavaOptions={java_opts_str}\" pyspark-shell"
    
    # Fallback for Hadoop home directory
    if "HADOOP_HOME" not in os.environ and os.name == 'nt':
        import tempfile
        hadoop_home = os.path.join(tempfile.gettempdir(), "hadoop_home")
        # Ensure directory structure exists
        os.makedirs(hadoop_home, exist_ok=True)
        os.makedirs(os.path.join(hadoop_home, "bin"), exist_ok=True)
        os.environ["HADOOP_HOME"] = hadoop_home
        # Also set hadoop.home.dir for Spark
        os.environ["hadoop.home.dir"] = hadoop_home
    
    # Set Python executable path for Spark workers (prevents Windows from redirecting to Microsoft Store)
    if os.name == 'nt':  # Windows
        if "PYSPARK_PYTHON" not in os.environ or "PYSPARK_DRIVER_PYTHON" not in os.environ:
            import sys
            python_exe = sys.executable  # Get the current Python executable path
            if not os.environ.get("PYSPARK_PYTHON"):
                os.environ["PYSPARK_PYTHON"] = python_exe
            if not os.environ.get("PYSPARK_DRIVER_PYTHON"):
                os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe
    
    # Kafka connector package for reading from Kafka
    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
    
    spark_builder = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "false") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.driver.extraJavaOptions", java_opts_str) \
        .config("spark.executor.extraJavaOptions", java_opts_str) \
        .config("spark.jars.packages", kafka_package) \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.python.worker.timeout", "1200s")
    
    # Configure Spark to handle missing winutils
    if os.name == 'nt':  # Windows
        spark_builder.config("spark.hadoop.fs.defaultFS", "file:///")
        # Disable some Hadoop features that require winutils
        spark_builder.config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    
    spark = spark_builder.getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def get_checkpoint_path(
    uploader,
    output_dir: str,
    checkpoint_prefix: str = "yolo_training"
) -> Optional[str]:
    """
    Get checkpoint path by downloading from MinIO if available
    
    Args:
        uploader: MinIOUploader instance
        output_dir: Output directory path
        checkpoint_prefix: Prefix for checkpoint objects in MinIO
    
    Returns:
        Path to checkpoint file if found, None otherwise
    """
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path_pt = checkpoint_dir / "last.pt"
    
    checkpoint_object_pt = f"{checkpoint_prefix}/last.pt"
    checkpoint_object_pth = f"{checkpoint_prefix}/last.pth"
    
    if uploader.checkpoint_exists(checkpoint_object_pt):
        print(f"Found existing checkpoint: {checkpoint_object_pt}")
        if uploader.download_checkpoint(str(checkpoint_path_pt), checkpoint_object_pt):
            return str(checkpoint_path_pt)
    elif uploader.checkpoint_exists(checkpoint_object_pth):
        print(f"Found checkpoint in .pth format: {checkpoint_object_pth}")
        checkpoint_path_pth = checkpoint_dir / "last.pth"
        if uploader.download_checkpoint(str(checkpoint_path_pth), checkpoint_object_pth):
            shutil.copy2(checkpoint_path_pth, checkpoint_path_pt)
            return str(checkpoint_path_pt)
    
    return None


def prepare_upload_files(
    output_dir: str,
    output_files: Dict[str, Path]
) -> Dict[str, Path]:
    """
    Prepare files for upload to MinIO
    
    Args:
        output_dir: Output directory path
        output_files: Dictionary of output files from trainer
    
    Returns:
        Dictionary of files ready for upload
    """
    from pathlib import Path
    
    train_dir = Path(output_dir) / "train" / "weights"
    upload_files = {}
    final_files = {k: v for k, v in output_files.items() if v and v.exists()}
    
    # Handle best weights - create .pth version
    best_pt = train_dir / "best.pt"
    if best_pt.exists():
        best_pth = train_dir / "best.pth"
        shutil.copy2(best_pt, best_pth)
        upload_files['best_weights'] = best_pth
    
    # Handle last weights
    last_pt = train_dir / "last.pt"
    if last_pt.exists():
        upload_files['last_weights'] = last_pt
    
    # Handle train log
    if 'train_log' in final_files:
        upload_files['train_log'] = final_files['train_log']
    elif (Path(output_dir) / "train" / "results.csv").exists():
        train_log_path = Path(output_dir) / "train" / "train.log"
        shutil.copy2(Path(output_dir) / "train" / "results.csv", train_log_path)
        upload_files['train_log'] = train_log_path
    
    # Handle config.yaml
    if 'config_yaml' in final_files and final_files['config_yaml'].exists():
        config_yaml_path = train_dir.parent / "config.yaml"
        shutil.copy2(final_files['config_yaml'], config_yaml_path)
        upload_files['config_yaml'] = config_yaml_path
    elif (train_dir.parent / "args.yaml").exists():
        config_yaml_path = train_dir.parent / "config.yaml"
        shutil.copy2(train_dir.parent / "args.yaml", config_yaml_path)
        upload_files['config_yaml'] = config_yaml_path
    
    # Handle metrics.json
    if 'metrics_json' in final_files:
        upload_files['metrics_json'] = final_files['metrics_json']
    
    return upload_files
