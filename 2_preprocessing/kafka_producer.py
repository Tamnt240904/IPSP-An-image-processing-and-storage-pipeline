import os
import time
import json
import lmdb
import msgpack
import datetime
import threading
from kafka import KafkaProducer
from minio import Minio

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
TOPIC_NAME = "traffic_data"
NUM_PARTITIONS = 10 

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "bigdataproject") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "bigdataproject")
BUCKET_NAME = "traffic-data"
MINIO_PREFIX = "raw_lmdb" 

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
            for _, value in cursor:
                try:
                    record = msgpack.unpackb(value, raw=False)
                    label_data = record['label']
                    label_data['timestamp'] = datetime.datetime.utcnow().isoformat()
                    label_data['camera_id'] = cam_id
                    
                    producer.send(
                        TOPIC_NAME, 
                        key=cam_id, 
                        value=label_data,
                        partition=target_partition # Ép đúng partition
                    )
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"📡 {cam_id} -> Partition {target_partition}: đã gửi {count} frames")
                    
                    time.sleep(1.0 / FPS)
                except Exception as e:
                    print(f"❌ Lỗi xử lý frame tại {cam_id}: {e}")
                    continue
        env.close()
        print(f"🏁 [Thread-{cam_id}] HOÀN THÀNH.")
    except Exception as e:
        print(f"⚠️ Lỗi thread {cam_id}: {e}")

def run_producer():
    print(f"🚀 [Producer] Khởi động Multi-threaded (Fixed Partitioning)...")
    
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        key_serializer=lambda k: k.encode('utf-8'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,
        batch_size=65536,
        linger_ms=10
    )

    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    local_paths = {}
    target_cams = get_target_cameras()
    
    for cam_id in target_cams:
        local_path = f"/tmp/{cam_id}.lmdb"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        
        found = False
        for f in ["data.mdb", "lock.mdb"]:
            try:
                minio_client.fget_object(BUCKET_NAME, f"{MINIO_PREFIX}/{cam_id}.lmdb/{f}", f"{local_path}/{f}")
                found = True
            except: continue
        if found:
            local_paths[cam_id] = local_path
            print(f"✅ Đã tải: {cam_id}")

    threads = []
    for cam_id, path in local_paths.items():
        t = threading.Thread(target=stream_single_camera, args=(cam_id, path, producer))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    producer.flush()
    print("🎯 TẤT CẢ CAMERA ĐÃ KẾT THÚC DỮ LIỆU.")

if __name__ == "__main__":
    run_producer()