import os
import time
import json
import lmdb
import pickle
import datetime
import threading
import base64
import numpy as np
from kafka import KafkaProducer
from minio import Minio

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
TOPIC_NAME = "traffic_data"
NUM_PARTITIONS = 10 

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "bigdataproject") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "bigdataproject")
BUCKET_NAME = "traffic-data"
MINIO_PREFIX = "lambda_lmdb"  # Đổi prefix mới

CAM_MODE = os.environ.get("CAM_MODE", "dev").lower()
FPS = 5 

def get_target_cameras():
    if CAM_MODE == "demo":
        return [f"cam_{i:02d}" for i in range(11, 21)]
    return [f"cam_{i:02d}" for i in range(1, 11)]

def stream_single_camera(cam_id, lmdb_path, producer):
    """Hàm chạy trong thread riêng cho từng camera"""
    print(f"🧵 [Thread-{cam_id}] Bắt đầu stream...")
    
    try:
        cam_num = int(cam_id.split('_')[1])
        target_partition = (cam_num - 1) % NUM_PARTITIONS
    except:
        target_partition = 0

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            count = 0
            for key, value in cursor:
                try:
                    # Giải mã key để lấy frame_id
                    frame_key = key.decode('utf-8')  # Ví dụ: 'cam_01_00001'
                    
                    # Unpack dữ liệu - thử pickle thay vì msgpack
                    try:
                        record = pickle.loads(value)
                    except Exception as e:
                        print(f"⚠️ Không thể unpickle, thử msgpack: {e}")
                        import msgpack
                        record = msgpack.unpackb(value, raw=False)
                    
                    # Lấy ảnh và encode sang base64
                    image_bytes = record.get('image')
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8') if image_bytes else None
                    
                    # Lấy boxes (numpy array) và convert sang list
                    boxes = record.get('boxes')
                    if boxes is not None:
                        # boxes shape: (N, 5) - [class_id, x_center, y_center, width, height]
                        boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
                    else:
                        boxes_list = []
                    
                    # Chuyển đổi boxes thành format objects (chỉ giữ thông tin cần thiết)
                    objects = []
                    for box in boxes_list:
                        obj = {
                            'class_id': int(box[0]),
                            'bbox': [float(box[1]), float(box[2]), float(box[3]), float(box[4])]  # [x_center, y_center, width, height]
                        }
                        objects.append(obj)
                    
                    # Tạo payload gửi vào Kafka (bao gồm image base64)
                    payload = {
                        'camera_id': cam_id,
                        'image_id': frame_key,
                        'image': image_base64,
                        'objects': objects
                    }
                    
                    # Gửi vào Kafka
                    producer.send(
                        TOPIC_NAME, 
                        key=cam_id, 
                        value=payload,
                        partition=target_partition
                    )
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"📡 {cam_id} -> Partition {target_partition}: đã gửi {count} frames")
                    
                    # Sleep để giữ FPS
                    time.sleep(1.0 / FPS)
                    
                except Exception as e:
                    print(f"❌ Lỗi xử lý frame tại {cam_id}: {e}")
                    continue
                    
        env.close()
        print(f"🏁 [Thread-{cam_id}] HOÀN THÀNH - Đã gửi {count} frames.")
        
    except Exception as e:
        print(f"⚠️ Lỗi thread {cam_id}: {e}")

def run_producer():
    print(f"🚀 [Producer] Khởi động Multi-threaded (Fixed Partitioning)...")
    print(f"📍 MINIO_PREFIX: {MINIO_PREFIX}")
    print(f"🎥 CAM_MODE: {CAM_MODE}")
    print(f"⚡ FPS: {FPS}")
    
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        key_serializer=lambda k: k.encode('utf-8'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,
        batch_size=65536,
        linger_ms=10
    )

    minio_client = Minio(
        MINIO_ENDPOINT, 
        access_key=MINIO_ACCESS_KEY, 
        secret_key=MINIO_SECRET_KEY, 
        secure=False
    )
    
    local_paths = {}
    target_cams = get_target_cameras()
    
    print(f"🔍 Đang tải dữ liệu từ MinIO bucket '{BUCKET_NAME}'...")
    
    for cam_id in target_cams:
        local_path = f"/tmp/{cam_id}.lmdb"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        
        found = False
        for f in ["data.mdb", "lock.mdb"]:
            try:
                object_path = f"{MINIO_PREFIX}/{cam_id}.lmdb/{f}"
                local_file = f"{local_path}/{f}"
                minio_client.fget_object(BUCKET_NAME, object_path, local_file)
                found = True
            except Exception as e:
                print(f"⚠️ Không tải được {object_path}: {e}")
                continue
                
        if found:
            local_paths[cam_id] = local_path
            print(f"✅ Đã tải: {cam_id}")
        else:
            print(f"❌ Không tìm thấy dữ liệu cho {cam_id}")

    if not local_paths:
        print("❌ Không có camera nào được tải về. Dừng producer.")
        return

    print(f"\n🚦 Bắt đầu stream {len(local_paths)} cameras...")
    
    threads = []
    for cam_id, path in local_paths.items():
        t = threading.Thread(target=stream_single_camera, args=(cam_id, path, producer))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    producer.flush()
    print("\n🎯 TẤT CẢ CAMERA ĐÃ KẾT THÚC DỮ LIỆU.")

if __name__ == "__main__":
    run_producer()