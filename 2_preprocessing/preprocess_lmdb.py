import os
import glob
import lmdb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCE_DATA_DIR = "/home/dell/Desktop/data_raw"
OUTPUT_LMDB_PATH = "/home/dell/Desktop/traffic_data.lmdb"
MAP_SIZE = 20 * 1024 * 1024 * 1024
NUM_THREADS = 4

def process_image(filepath):
    try:
        filename = os.path.basename(filepath)
        record_key = filename.rsplit('.', 1)[0]
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        return (record_key, image_bytes)
    except Exception as e:
        return None

def create_lmdb_optimized(source_dir, lmdb_path):
    if not os.path.exists(source_dir):
        print(f"Lỗi: Không tìm thấy {source_dir}")
        return

    image_files = glob.glob(os.path.join(source_dir, "*.jpg"))
    print(f"--- BẮT ĐẦU TẠO LMDB (ĐA LUỒNG: {NUM_THREADS} threads) ---")
    print(f"Tổng số ảnh: {len(image_files)}")

    env = lmdb.open(lmdb_path, map_size=MAP_SIZE)
    txn = env.begin(write=True)
    
    cnt = 0
    batch_size = 1000

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_image, f): f for f in image_files}

        for future in tqdm(as_completed(futures), total=len(image_files), desc="Writing LMDB"):
            result = future.result()
            if result:
                key, val = result
                txn.put(key=key.encode('ascii'), value=val)
                cnt += 1

                if cnt % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
    
    txn.commit()
    env.close()
    print(f"\n✅ XONG! Đã xử lý {cnt} ảnh.")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_LMDB_PATH):
        print("File LMDB cũ đã tồn tại, đang xóa để tạo mới...")
        import shutil
        shutil.rmtree(OUTPUT_LMDB_PATH, ignore_errors=True)
        
    create_lmdb_optimized(SOURCE_DATA_DIR, OUTPUT_LMDB_PATH)
