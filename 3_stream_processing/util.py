import cv2
import numpy as np
import json
import math
import os
from datetime import datetime

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER", "bigdataproject")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD", "bigdataproject")
BUCKET_CONFIGS = "configs"

_road_masks_cache = {}

_prev_frame_state = {}

IMG_W, IMG_H = 1280, 720
PIXELS_PER_METER = 20.0 
MAX_MATCH_DIST = 150.0   
def get_minio_client():
    from minio import Minio
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

def load_road_mask(camera_id, h=720, w=1280):
    if camera_id in _road_masks_cache:
        return _road_masks_cache[camera_id]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        client = get_minio_client()
        mapping = {
            "cam_01_02_03_11_12_13.json": ["cam_01", "cam_02", "cam_03", "cam_11", "cam_12", "cam_13"],
            "cam_04_05_14_15.json": ["cam_04", "cam_05", "cam_14", "cam_15"],
            "cam_06_07_08_16_17_18.json": ["cam_06", "cam_07", "cam_08", "cam_16", "cam_17", "cam_18"],
            "cam_09_19.json": ["cam_09", "cam_19"],
            "cam_10_20.json": ["cam_10", "cam_20"]
        }
        target_file = next((f for f, cams in mapping.items() if camera_id in cams), None)
        if target_file:
            response = client.get_object(BUCKET_CONFIGS, f"road_geometry/{target_file}")
            config_data = json.loads(response.read().decode('utf-8'))
            for shape in config_data.get("shapes", []):
                if shape.get("label") in ["road", camera_id]:
                    pts = np.array(shape["points"], np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        
        _road_masks_cache[camera_id] = mask
    except Exception as e:
        print(f"Error loading mask for {camera_id}: {e}")
        return np.ones((h, w), dtype=np.uint8)
    return mask

def process_traffic_analysis(camera_id, timestamp_str, objects):
    global _prev_frame_state
    
    try:
        try:
            ts_clean = timestamp_str.replace('Z', '+00:00')
            t_curr = datetime.fromisoformat(ts_clean).timestamp()
        except:
            t_curr = datetime.now().timestamp()

        if not objects:
            return json.dumps({
                "camera_id": camera_id, "timestamp": timestamp_str,
                "density": {"percentage": 0.0},
                "unique_counts": {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0},
                "avg_speeds_frame": {"avg_speed_motorcycle": 0.0, "avg_speed_car": 0.0, "avg_speed_bus": 0.0, "avg_speed_truck": 0.0}
            })

        road_mask = load_road_mask(camera_id, IMG_H, IMG_W)
        road_pixel_count = cv2.countNonZero(road_mask)
        vehicle_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        
        label_map = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
        unique_counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        speeds_list = {"motorcycle": [], "car": [], "bus": [], "truck": []}
        
        # Lấy dữ liệu frame trước để so sánh
        prev_data = _prev_frame_state.get(camera_id)
        prev_objects = prev_data['objects'] if prev_data else []
        t_prev = prev_data['timestamp'] if prev_data else 0
        dt = t_curr - t_prev
        
        if dt > 2.0 or dt <= 0:
            prev_objects = []
            dt = 0

        curr_frame_objects_save = [] 
        for row_obj in objects:
            obj = row_obj.asDict() if hasattr(row_obj, "asDict") else row_obj
            cls_id = obj.get('class_id')
            bbox = obj.get('bbox')
            
            cls_name = label_map.get(cls_id, "unknown")

            if bbox and len(bbox) == 4:
                nx, ny, nw, nh = bbox
                w_px, h_px = int(nw * IMG_W), int(nh * IMG_H)
                cx_px, cy_px = int(nx * IMG_W), int(ny * IMG_H)
                
                x1 = int(cx_px - w_px/2)
                y1 = int(cy_px - h_px/2)
                x2 = x1 + w_px
                y2 = y1 + h_px
                
                cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 1, -1)
                
                curr_obj_data = {"class_id": cls_id, "center": (cx_px, cy_px)}
                curr_frame_objects_save.append(curr_obj_data)
                
                speed_kmh = 0.0
                matched = False
                
                if prev_objects and dt > 0:
                    min_dist = float('inf')
                    best_match_idx = -1
                    
                    for idx, p_obj in enumerate(prev_objects):
                        if p_obj['class_id'] == cls_id:
                            dist = math.sqrt((cx_px - p_obj['center'][0])**2 + (cy_px - p_obj['center'][1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_match_idx = idx
                    
                    if min_dist < MAX_MATCH_DIST and best_match_idx != -1:
                        matched = True
                        dist_meters = min_dist / PIXELS_PER_METER
                        speed_mps = dist_meters / dt
                        speed_kmh = speed_mps * 3.6
                        
                        prev_objects.pop(best_match_idx)

                if matched:
                    if cls_name in speeds_list:
                        speeds_list[cls_name].append(speed_kmh)
                else:
                    if cls_name in unique_counts:
                        unique_counts[cls_name] += 1
                    if cls_name in speeds_list:
                        speeds_list[cls_name].append(0.0)

        overlap = cv2.bitwise_and(vehicle_mask, road_mask)
        density_pct = (cv2.countNonZero(overlap) / road_pixel_count * 100) if road_pixel_count > 0 else 0
        
        avg_speeds = {f"avg_speed_{k}": (sum(v)/len(v) if v else 0.0) for k, v in speeds_list.items()}
        
        _prev_frame_state[camera_id] = {
            "timestamp": t_curr,
            "objects": curr_frame_objects_save
        }

        return json.dumps({
            "camera_id": camera_id,
            "timestamp": timestamp_str,
            "density": {"percentage": round(density_pct, 2)},
            "unique_counts": unique_counts,
            "avg_speeds_frame": avg_speeds
        })

    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e), "timestamp": timestamp_str})