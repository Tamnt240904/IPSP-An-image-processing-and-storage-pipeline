import time
import json
import glob
import os
from datetime import datetime
from kafka import KafkaProducer

DATA_DIR = "/home/dell/Desktop/data_raw" 
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC_NAME = 'traffic-metadata'

def create_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=16384, 
            linger_ms=10
        )
        print(f"✅ Đã kết nối tới Kafka tại {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        print(f"❌ Lỗi kết nối Kafka: {e}")
        return None

def get_file_timestamp(filepath):
    try:
        timestamp = os.path.getmtime(filepath)
        return datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        return None

def generate_message(filepath):
    filename = os.path.basename(filepath)
    record_key = filename.rsplit('.', 1)[0]
    
    parts = record_key.split('_')
    if len(parts) >= 2:
        camera_id = f"{parts[0]}_{parts[1]}"
    else:
        camera_id = "cam_unknown"

    timestamp = get_file_timestamp(filepath)

    message = {
        "record_key": record_key,
        "camera_id": camera_id,
        "lmdb_info": {
            "lmdb_filepath": "traffic-data/lmdb_data/data.mdb",
            "frame_height": 720,
            "frame_width": 1280
        },
        "schema_version": "1.0"
    }

    if timestamp:
        message["timestamp"] = timestamp

    return message

def run_batch_producer():
    print("📂 Đang quét danh sách ảnh...")
    jpg_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    jpg_files.sort()
    
    total_files = len(jpg_files)
    if total_files == 0:
        print("❌ Không tìm thấy ảnh nào! Kiểm tra lại đường dẫn DATA_DIR.")
        return

    print(f"👉 Tìm thấy {total_files} ảnh. Bắt đầu chế độ BATCH INGESTION (Xả lũ)...")

    producer = create_producer()
    if not producer:
        return

    count = 0
    start_time = time.time()

    try:
        for filepath in jpg_files:
            msg = generate_message(filepath)
            
            producer.send(TOPIC_NAME, key=msg['record_key'].encode('utf-8'), value=msg)
            
            count += 1
            
            if count % 1000 == 0:
                print(f"🚀 Đã đẩy {count}/{total_files} bản tin...")

        producer.flush()
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✅ HOÀN TẤT! Đã gửi {count} bản tin.")
        print(f"⏱️ Thời gian chạy: {duration:.2f} giây.")
        print(f"⚡ Tốc độ trung bình: {count/duration:.0f} tin/giây.")

    except KeyboardInterrupt:
        print("\n🛑 Đã dừng bởi người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi runtime: {e}")
    finally:
        producer.close()

if __name__ == "__main__":
    run_batch_producer()