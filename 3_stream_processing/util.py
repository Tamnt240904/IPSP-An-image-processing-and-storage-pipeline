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
_track_history = {}
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
    except:
        return np.ones((h, w), dtype=np.uint8)
    return mask
def calculate_speed_kmh(camera_id, track_id, current_pos, timestamp_str):
    global _track_history
    if track_id is None: return 0.0
    combined_key = f"{camera_id}_{track_id}"
    try:
        t_curr = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
    except:
        t_curr = datetime.now().timestamp()
    speed_kmh = 0.0
    curr_x, curr_y = current_pos.get('x', 0), current_pos.get('y', 0)
    if combined_key in _track_history:
        prev = _track_history[combined_key]
        dt = t_curr - prev['t']
        if 0 < dt < 2.0:
            dist = math.sqrt((curr_x - prev['x'])**2 + (curr_y - prev['y'])**2)
            speed_kmh = (dist / dt) * 3.6
    _track_history[combined_key] = {'x': curr_x, 'y': curr_y, 't': t_curr}
    return float(speed_kmh)
_seen_tracks = set()
def process_traffic_analysis(camera_id, timestamp, objects):
    global _seen_tracks
    try:
        h, w = 720, 1280
        if not objects:
            return json.dumps({"camera_id": camera_id, "status": "no_data", "timestamp": timestamp})
        road_mask = load_road_mask(camera_id, h, w)
        road_pixel_count = cv2.countNonZero(road_mask)
        vehicle_mask = np.zeros((h, w), dtype=np.uint8)
        label_map = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
        unique_counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        speeds_by_type = {"motorcycle": [], "car": [], "bus": [], "truck": []}
        world_coords = []
        all_polygons = []
        for row_obj in objects:
            obj = row_obj.asDict(recursive=True) if hasattr(row_obj, "asDict") else row_obj
            cls_name = label_map.get(obj.get('class_id'), "unknown")
            track_id = obj.get('track_id')
            combined_id = f"{camera_id}_{track_id}"
            if combined_id not in _seen_tracks:
                if cls_name in unique_counts:
                    unique_counts[cls_name] += 1
                _seen_tracks.add(combined_id)
            for poly in obj.get('segmentation', []):
                if poly: all_polygons.append(np.array(poly, np.int32))
            curr_pos = obj.get('world_coordinates', {'x': 0, 'y': 0})
            speed = calculate_speed_kmh(camera_id, track_id, curr_pos, timestamp)
            if cls_name in speeds_by_type:
                speeds_by_type[cls_name].append(speed)
            world_coords.append([curr_pos.get('x', 0), curr_pos.get('y', 0)])
        if all_polygons: cv2.fillPoly(vehicle_mask, all_polygons, 1)
        overlap = cv2.bitwise_and(vehicle_mask, road_mask)
        density_pct = (cv2.countNonZero(overlap) / road_pixel_count * 100) if road_pixel_count > 0 else 0
        avg_speeds = {f"avg_speed_{k}": (sum(v)/len(v) if v else 0.0) for k, v in speeds_by_type.items()}
        return json.dumps({
            "camera_id": camera_id,
            "timestamp": timestamp,
            "density": {"percentage": round(density_pct, 2)},
            "unique_counts": unique_counts,
            "avg_speeds_frame": avg_speeds
        })
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)})