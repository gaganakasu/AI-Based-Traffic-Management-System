import os
import cv2
import numpy as np
import base64
import math
from typing import Dict, Any, List, Optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Target vehicle class IDs in COCO dataset
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_MAP = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Density threshold rules (Configurable prototype parameters)
DENSITY_THRESHOLDS = {
    "LOW_MAX": 5,      # 0-5 vehicles -> LOW -> 15s
    "MEDIUM_MAX": 15,  # 6-15 vehicles -> MEDIUM -> 30s
                       # >15 vehicles -> HIGH -> 45s
}

SIGNAL_TIMING_RULES = {
    "LOW": 15,
    "MEDIUM": 30,
    "HIGH": 45
}

class YOLOVehicleDetector:
    _instance = None

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if YOLO_AVAILABLE:
            try:
                # Load lightweight pretrained model
                self.model = YOLO(self.model_name)
                self.is_loaded = True
                print(f"[YOLOVehicleDetector] Loaded {self.model_name} successfully.")
            except Exception as e:
                print(f"[YOLOVehicleDetector] Could not load YOLO model: {e}")
                self.is_loaded = False
        else:
            print("[YOLOVehicleDetector] Ultralytics YOLO package not available.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = YOLOVehicleDetector()
        return cls._instance

    def evaluate_traffic_density(self, vehicle_count: int) -> Dict[str, Any]:
        """
        Applies prototype density & signal recommendation rules:
        - Low (0–5 vehicles): 15 seconds
        - Medium (6–15 vehicles): 30 seconds
        - High (>15 vehicles): 45 seconds
        """
        if vehicle_count <= DENSITY_THRESHOLDS["LOW_MAX"]:
            density = "LOW"
            recommended_green = SIGNAL_TIMING_RULES["LOW"]
            recommendation_text = f"Low traffic density ({vehicle_count} vehicles). Recommended green signal duration: {recommended_green}s to avoid idle delays."
        elif vehicle_count <= DENSITY_THRESHOLDS["MEDIUM_MAX"]:
            density = "MEDIUM"
            recommended_green = SIGNAL_TIMING_RULES["MEDIUM"]
            recommendation_text = f"Moderate traffic density ({vehicle_count} vehicles). Recommended green signal duration: {recommended_green}s for balanced throughput."
        else:
            density = "HIGH"
            recommended_green = SIGNAL_TIMING_RULES["HIGH"]
            recommendation_text = f"High traffic density ({vehicle_count} vehicles). Recommended green signal duration extended to {recommended_green}s to clear bottleneck."

        return {
            "density_level": density,
            "recommended_green_seconds": recommended_green,
            "recommendation_text": recommendation_text,
            "thresholds": DENSITY_THRESHOLDS
        }

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Runs YOLO inference on a single BGR OpenCV frame, filters vehicle classes,
        draws bounding boxes, and returns vehicle metrics.
        """
        counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        detections = []
        annotated_frame = frame.copy()

        if self.is_loaded and self.model is not None:
            try:
                results = self.model(frame, conf=conf_threshold, verbose=False)[0]
                
                # Colors for bounding boxes (BGR)
                color_map = {
                    "Car": (255, 240, 0),        # Cyan
                    "Motorcycle": (247, 85, 168), # Purple
                    "Bus": (0, 215, 255),        # Yellow/Gold
                    "Truck": (127, 0, 255)       # Pink/Red
                }

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in VEHICLE_CLASS_MAP:
                        cls_name = VEHICLE_CLASS_MAP[cls_id]
                        counts[cls_name] += 1

                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy

                        color = color_map.get(cls_name, (0, 255, 135))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label text
                        label = f"{cls_name} {conf:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                        detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 3),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
            except Exception as e:
                print(f"[YOLOVehicleDetector] Inference error: {e}")

        total_vehicles = sum(counts.values())
        evaluation = self.evaluate_traffic_density(total_vehicles)

        # Add HUD overlay
        hud_bg = (15, 23, 42)
        cv2.rectangle(annotated_frame, (10, 10), (320, 95), hud_bg, -1)
        cv2.rectangle(annotated_frame, (10, 10), (320, 95), (0, 240, 255), 1)
        cv2.putText(annotated_frame, f"YOLOv8 AI Vehicle Detection", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Total Vehicles: {total_vehicles} | Density: {evaluation['density_level']}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Cars: {counts['Car']} | Moto: {counts['Motorcycle']} | Bus: {counts['Bus']} | Truck: {counts['Truck']}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Rec. Green: {evaluation['recommended_green_seconds']}s", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 135), 1, cv2.LINE_AA)

        return {
            "total_vehicles": total_vehicles,
            "counts": counts,
            "detections": detections,
            "density_level": evaluation["density_level"],
            "recommended_green_seconds": evaluation["recommended_green_seconds"],
            "recommendation_text": evaluation["recommendation_text"],
            "annotated_frame": annotated_frame
        }

    def analyze_video_file(self, video_path: str, max_frames: int = 75, sample_interval: int = 2) -> Dict[str, Any]:
        """
        Processes video file, samples frames, runs YOLO vehicle detection,
        and generates summary metrics + representative annotated snapshot images (Base64).
        """
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video stream"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_results = []
        annotated_snapshots = []
        frame_idx = 0
        processed_count = 0

        max_vehicles_seen = 0
        total_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                result = self.process_frame(frame)
                
                # Aggregate counts
                for k, v in result["counts"].items():
                    total_counts[k] = max(total_counts[k], v)

                max_vehicles_seen = max(max_vehicles_seen, result["total_vehicles"])

                frame_results.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "total_vehicles": result["total_vehicles"],
                    "counts": result["counts"],
                    "density_level": result["density_level"]
                })

                # Save 4 representative snapshot thumbnails as base64
                if processed_count in [0, 10, 25, 45] or (processed_count == max_frames - 1):
                    _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
                    b64_str = base64.b64encode(buffer).decode("utf-8")
                    annotated_snapshots.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "vehicles": result["total_vehicles"],
                        "density": result["density_level"],
                        "image_base64": f"data:image/jpeg;base64,{b64_str}"
                    })

                processed_count += 1

            frame_idx += 1

        cap.release()

        # Compute averages & overall evaluation
        avg_vehicles = round(sum(f["total_vehicles"] for f in frame_results) / len(frame_results), 1) if frame_results else 0
        overall_eval = self.evaluate_traffic_density(int(round(avg_vehicles)))

        return {
            "status": "success",
            "model_used": "YOLOv8n (Pretrained on COCO Vehicle Classes)",
            "frames_analyzed": processed_count,
            "total_video_frames": total_video_frames,
            "fps": fps,
            "average_vehicles_detected": avg_vehicles,
            "peak_vehicles_detected": max_vehicles_seen,
            "vehicle_category_breakdown": total_counts,
            "traffic_density": overall_eval["density_level"],
            "recommended_green_seconds": overall_eval["recommended_green_seconds"],
            "ai_signal_recommendation": overall_eval["recommendation_text"],
            "timeline": frame_results[::3],  # Subsampled for chart
            "snapshots": annotated_snapshots
        }

    @staticmethod
    def generate_demo_traffic_video(output_path: str, duration_sec: int = 5, fps: int = 25):
        """
        Creates a synthetic demo traffic video with multiple lanes, moving cars, buses,
        and trucks for zero-setup demo testing.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = duration_sec * fps
        
        # Vehicles with start positions, speeds, dimensions, colors
        vehicles = [
            {"type": "Car", "x": 120, "y": 0, "speed": 4, "w": 40, "h": 70, "color": (50, 50, 220)},
            {"type": "Car", "x": 125, "y": -180, "speed": 4.2, "w": 38, "h": 68, "color": (220, 100, 50)},
            {"type": "Bus", "x": 180, "y": -50, "speed": 3, "w": 52, "h": 120, "color": (50, 180, 240)},
            {"type": "Truck", "x": 250, "y": -220, "speed": 3.5, "w": 55, "h": 130, "color": (100, 100, 100)},
            {"type": "Motorcycle", "x": 320, "y": -100, "speed": 5.5, "w": 20, "h": 40, "color": (180, 50, 220)},
            {"type": "Car", "x": 370, "y": 500, "speed": -4.5, "w": 40, "h": 70, "color": (60, 200, 80)},
            {"type": "Car", "x": 440, "y": 620, "speed": -4, "w": 40, "h": 70, "color": (240, 200, 50)},
            {"type": "Truck", "x": 500, "y": 550, "speed": -3.2, "w": 54, "h": 125, "color": (150, 80, 50)},
        ]

        for f in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Road background (dark asphalt)
            frame[:] = (40, 44, 52)

            # Road borders and lanes
            cv2.rectangle(frame, (80, 0), (560, height), (30, 32, 38), -1)
            cv2.line(frame, (80, 0), (80, height), (255, 255, 255), 2)
            cv2.line(frame, (560, 0), (560, height), (255, 255, 255), 2)

            # Center dividing line (Double Yellow)
            cv2.line(frame, (318, 0), (318, height), (0, 215, 255), 2)
            cv2.line(frame, (322, 0), (322, height), (0, 215, 255), 2)

            # Dashed lane dividers
            dash_offset = (f * 5) % 40
            for y in range(-40 + dash_offset, height + 40, 40):
                cv2.line(frame, (160, y), (160, y + 20), (200, 200, 200), 1)
                cv2.line(frame, (240, y), (240, y + 20), (200, 200, 200), 1)
                cv2.line(frame, (400, y), (400, y + 20), (200, 200, 200), 1)
                cv2.line(frame, (480, y), (480, y + 20), (200, 200, 200), 1)

            # Draw vehicles
            for v in vehicles:
                curr_y = int(v["y"] + (f * v["speed"]))
                # Loop around
                if v["speed"] > 0 and curr_y > height + 100:
                    curr_y = -150 + (curr_y % (height + 250))
                elif v["speed"] < 0 and curr_y < -150:
                    curr_y = height + 100 - (abs(curr_y) % (height + 250))

                vx, vy, vw, vh = v["x"], curr_y, v["w"], v["h"]

                # Vehicle body
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), v["color"], -1, cv2.LINE_AA)
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (255, 255, 255), 1, cv2.LINE_AA)

                # Windshields & Wheels
                if v["type"] in ["Car", "Bus", "Truck"]:
                    cv2.rectangle(frame, (vx + 4, vy + 8), (vx + vw - 4, vy + 22), (20, 20, 30), -1)
                    cv2.rectangle(frame, (vx + 4, vy + vh - 20), (vx + vw - 4, vy + vh - 8), (20, 20, 30), -1)

                # Headlights
                if v["speed"] > 0:
                    cv2.circle(frame, (vx + 6, vy + vh - 3), 3, (200, 255, 255), -1)
                    cv2.circle(frame, (vx + vw - 6, vy + vh - 3), 3, (200, 255, 255), -1)
                else:
                    cv2.circle(frame, (vx + 6, vy + 3), 3, (200, 255, 255), -1)
                    cv2.circle(frame, (vx + vw - 6, vy + 3), 3, (200, 255, 255), -1)

            # Overlay simulated CCTV Timestamp & Camera Name
            cv2.putText(frame, "CAM-04 (NORTH ARTERIAL APPROACH) [LIVE FEED SIMULATION]", (90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)
            
            out.write(frame)

        out.release()
        print(f"[YOLOVehicleDetector] Generated sample traffic video at {output_path}")
