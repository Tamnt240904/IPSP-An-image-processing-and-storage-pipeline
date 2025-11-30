import lmdb
import cv2
import numpy as np
import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

LOCAL_LMDB_PATH = "/home/dell/GitHub/IPSP-An-image-processing-and-storage-pipeline/lmdb_data"
LOCAL_MODEL_PATH = "/home/dell/GitHub/IPSP-An-image-processing-and-storage-pipeline/4_model_training/model/yolov8_traffic_best.pt"

_yolo_model = None
_mask_processor = None
_mask_model = None
_lmdb_env = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(LOCAL_MODEL_PATH):
            print(f"⚠️ Không tìm thấy model tại {LOCAL_MODEL_PATH}, đang dùng yolov8m.pt mặc định.")
            _yolo_model = YOLO("yolov8m.pt")
        else:
            print(f"🔄 [Init] Đang load YOLO từ: {LOCAL_MODEL_PATH}")
            _yolo_model = YOLO(LOCAL_MODEL_PATH)
    return _yolo_model

def get_mask_model():
    global _mask_processor, _mask_model
    if _mask_model is None:
        model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
        print(f"🚀 [Init] Đang load Mask2Former ({_device})...")
        _mask_processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        _mask_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        _mask_model.to(_device)
        _mask_model.eval()
    return _mask_processor, _mask_model

def get_lmdb_env():
    global _lmdb_env
    if _lmdb_env is None:
        if os.path.exists(LOCAL_LMDB_PATH):
            _lmdb_env = lmdb.open(LOCAL_LMDB_PATH, readonly=True, lock=False)
        else:
            print(f"❌ Không tìm thấy LMDB tại: {LOCAL_LMDB_PATH}")
            return None
    return _lmdb_env

def get_road_mask(image_numpy):
    processor, model = get_mask_model()
    
    image_pil = Image.fromarray(image_numpy)
    inputs = processor(images=image_pil, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_pil.size[::-1]]
    )[0].cpu().numpy()
    
    raw_mask = (pred_map == 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(raw_mask)

    largest_contour = max(contours, key=cv2.contourArea)
    processed_mask = np.zeros_like(raw_mask)
    cv2.drawContours(processed_mask, [largest_contour], -1, 1, -1)
    
    kernel = np.ones((7,7), np.uint8) 
    processed_mask = cv2.dilate(processed_mask, kernel, iterations=3)
    
    return processed_mask

def process_image_logic(record_key):
    try:
        env = get_lmdb_env()
        if env is None: return {"error": "LMDB missing", "status": "failed"}

        with env.begin() as txn:
            img_bytes = txn.get(record_key.encode('ascii'))
            
        if img_bytes is None: return {"error": "Key missing", "status": "failed"}

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None: return {"error": "Decode failed", "status": "failed"}
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        road_mask = get_road_mask(img_rgb)
        road_area_empty = cv2.countNonZero(road_mask)

        yolo_model = get_yolo_model()
        results = yolo_model(img_bgr, verbose=False)
        result = results[0]

        mask_vehicles = np.zeros((h, w), dtype=np.uint8)
        class_counts = {}

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(mask_vehicles, (x1, y1), (x2, y2), 255, -1)
            
            cls_id = int(box.cls[0])
            if hasattr(yolo_model.names, 'get'):
                cls_name = yolo_model.names.get(cls_id, str(cls_id))
            else:
                cls_name = str(cls_id)
                
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        vehicle_area = cv2.countNonZero(mask_vehicles)

        total_road_surface = vehicle_area + road_area_empty
        
        if total_road_surface > 0:
            density = round(vehicle_area / total_road_surface, 4)
        else:
            density = 0.0

        status_traffic = "Normal"
        if density < 0.2: status_traffic = "Free"
        elif density > 0.5: status_traffic = "Congested"

        return {
            "counts": class_counts,
            "density": density,
            "status": "success",
            "traffic_status": status_traffic,
            "metrics_debug": {
                "road_pixels": int(road_area_empty),
                "vehicle_pixels": int(vehicle_area)
            }
        }

    except Exception as e:
        return {
            "counts": {},
            "density": 0.0,
            "status": "failed",
            "traffic_status": "Error",
            "metrics_debug": {},
            "error": str(e)
        }

if __name__ == "__main__":
    TEST_KEY = "cam_01_00544" 
    print(f"🛠️  Đang chạy test local udf_logic.py với key: {TEST_KEY}")
    
    result = process_image_logic(TEST_KEY)
    
    print("\n✅ KẾT QUẢ TEST:")
    print(result)