"""
MinIO uploader module for uploading training outputs
"""
from pathlib import Path
from typing import Dict, Optional
from minio import Minio
from minio.error import S3Error


class MinIOUploader:
    """Upload training outputs to MinIO"""
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False
    ):
        """
        Initialize MinIO uploader
        
        Args:
            endpoint: MinIO endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket: Bucket name
            secure: Use secure connection (HTTPS)
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket = bucket
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if not"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                print(f"✓ Created bucket: {self.bucket}")
            else:
                print(f"✓ Bucket {self.bucket} already exists")
        except S3Error as e:
            print(f"✗ ERROR: Failed to ensure bucket exists: {self.bucket}")
            print(f"  Error: {e}")
            raise
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR: Failed to connect to MinIO")
            print(f"  Endpoint: {self.client._base_url}")
            print(f"  Error: {e}")
            raise
    
    def upload_file(self, local_path: str, object_name: str) -> bool:
        """
        Upload a single file to MinIO
        
        Args:
            local_path: Local file path
            object_name: Object name in MinIO
            
        Returns:
            True if successful, False otherwise
        """
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                print(f"ERROR: File not found: {local_path}")
                return False
            
            file_size = local_file.stat().st_size
            print(f"Uploading {object_name} ({file_size / (1024*1024):.2f} MB)...")
            
            self.client.fput_object(
                self.bucket,
                object_name,
                str(local_file)
            )
            print(f"✓ Successfully uploaded: {object_name}")
            return True
        except S3Error as e:
            print(f"✗ ERROR uploading {object_name}: {e}")
            print(f"  Error code: {e.code if hasattr(e, 'code') else 'unknown'}")
            print(f"  Error message: {e.message if hasattr(e, 'message') else str(e)}")
            return False
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR uploading {object_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upload_training_outputs(
        self,
        files: Dict[str, Path],
        prefix: str = "yolo_training"
    ) -> Dict[str, bool]:
        """
        Upload all training output files
        
        Args:
            files: Dictionary with file type as key and Path as value
            prefix: Prefix for object names in MinIO
            
        Returns:
            Dictionary with file type as key and upload success status as value
        """
        results = {}
        file_mapping = {
            'best_weights': f"{prefix}/best.pth",
            'last_weights': f"{prefix}/last.pt",
            'train_log': f"{prefix}/train.log",
            'config_yaml': f"{prefix}/config.yaml",
            'metrics_json': f"{prefix}/metrics.json"
        }
        
        for file_type, local_path in files.items():
            if local_path and local_path.exists():
                object_name = file_mapping.get(file_type, f"{prefix}/{local_path.name}")
                
                # For results.csv, rename to train.log
                if file_type == 'train_log' and local_path.name == 'results.csv':
                    import shutil
                    train_log_path = local_path.parent / "train.log"
                    shutil.copy2(local_path, train_log_path)
                    success = self.upload_file(str(train_log_path), object_name)
                else:
                    success = self.upload_file(str(local_path), object_name)
                
                results[file_type] = success
            else:
                print(f"File not found for {file_type}: {local_path}")
                results[file_type] = False
        
        return results
    
    def upload_dataset_yaml(self, yaml_path: str, prefix: str = "yolo_training") -> bool:
        """
        Upload dataset YAML file
        
        Args:
            yaml_path: Path to dataset YAML file
            prefix: Prefix for object name
            
        Returns:
            True if successful, False otherwise
        """
        object_name = f"{prefix}/dataset.yaml"
        return self.upload_file(yaml_path, object_name)
    
    def download_checkpoint(self, local_path: str, object_name: str) -> bool:
        """
        Download checkpoint from MinIO
        
        Args:
            local_path: Local file path to save the checkpoint
            object_name: Object name in MinIO
            
        Returns:
            True if successful, False otherwise
        """
        try:
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.client.fget_object(
                self.bucket,
                object_name,
                str(local_file)
            )
            print(f"Downloaded checkpoint: {object_name} -> {local_path}")
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                print(f"Checkpoint not found: {object_name}")
            else:
                print(f"Error downloading checkpoint {object_name}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error downloading checkpoint {object_name}: {e}")
            return False
    
    def checkpoint_exists(self, object_name: str) -> bool:
        """
        Check if checkpoint exists in MinIO
        
        Args:
            object_name: Object name in MinIO
            
        Returns:
            True if exists, False otherwise
        """
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            raise
        except Exception:
            return False
    
    def save_training_state(self, state: dict, object_name: str = "yolo_training/training_state.json") -> bool:
        """Save training state to MinIO"""
        try:
            import json
            import tempfile
            from pathlib import Path
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(state, f, indent=2)
                temp_path = f.name
            
            success = self.upload_file(temp_path, object_name)
            Path(temp_path).unlink()
            return success
        except Exception as e:
            print(f"Error saving training state: {e}")
            return False
    
    def load_training_state(self, object_name: str = "yolo_training/training_state.json") -> Optional[dict]:
        """Load training state from MinIO"""
        try:
            import json
            import tempfile
            from pathlib import Path
            
            if not self.checkpoint_exists(object_name):
                return None
            
            with tempfile.NamedTemporaryFile(mode='r', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            self.client.fget_object(self.bucket, object_name, temp_path)
            with open(temp_path, 'r') as f:
                state = json.load(f)
            Path(temp_path).unlink()
            return state
        except Exception as e:
            print(f"Error loading training state: {e}")
            return None

