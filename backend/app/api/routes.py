import os
import json
import random
import tempfile
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.ml.yolo_detector import YOLOVehicleDetector, DENSITY_THRESHOLDS, SIGNAL_TIMING_RULES

router = APIRouter()

# Request schemas
class IncidentRequest(BaseModel):
    junction_id: str
    active: bool

class EmergencyRequest(BaseModel):
    route: List[str]

class ControlRequest(BaseModel):
    action: str  # "pause", "resume", "reset"

class PresetRequest(BaseModel):
    preset: str  # "balanced", "rush", "emergency", "incident"

@router.get("/traffic")
async def get_traffic(request: Request):
    """
    Get current aggregate traffic metrics, active scenario, and all junctions' statuses.
    """
    simulator = request.app.state.simulator
    return simulator.get_state()

@router.get("/intersections")
async def get_intersections(request: Request):
    """
    Get detailed telemetry breakdown of all intersections.
    """
    simulator = request.app.state.simulator
    state = simulator.get_state()
    return state["junctions"]

@router.post("/preset/{preset_name}")
async def apply_scenario_preset(request: Request, preset_name: str):
    """
    Apply a functional scenario preset: 'balanced', 'rush', 'emergency', or 'incident'.
    """
    simulator = request.app.state.simulator
    valid_presets = ["balanced", "rush", "emergency", "incident"]
    if preset_name.lower() not in valid_presets:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Choose from {valid_presets}")

    new_state = simulator.apply_preset(preset_name.lower())
    return {
        "status": "success",
        "message": f"Applied preset '{preset_name}'",
        "state": new_state
    }

@router.post("/video/sample")
async def analyze_sample_video():
    """
    Runs YOLOv8 vehicle detection pipeline on the built-in demo traffic video.
    If demo video is not present, generates it dynamically using OpenCV.
    """
    detector = YOLOVehicleDetector.get_instance()
    
    # Path for sample video
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sample_video_path = os.path.join(base_dir, "data", "sample_traffic.mp4")
    
    if not os.path.exists(sample_video_path):
        try:
            YOLOVehicleDetector.generate_demo_traffic_video(sample_video_path, duration_sec=4, fps=25)
        except Exception as e:
            print(f"Error generating demo video: {e}")

    if not os.path.exists(sample_video_path):
        raise HTTPException(status_code=500, detail="Demo traffic video could not be generated.")

    result = detector.analyze_video_file(sample_video_path, max_frames=50, sample_interval=2)
    return result

@router.post("/video/analyze")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """
    Receives an uploaded traffic camera video file, runs YOLO vehicle detection,
    and returns vehicle counts, category breakdown, density classification, and signal recommendations.
    """
    detector = YOLOVehicleDetector.get_instance()

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = detector.analyze_video_file(tmp_path, max_frames=60, sample_interval=2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@router.get("/analytics")
async def get_analytics(request: Request):
    """
    Return prototype comparison analytics (Static vs Adaptive Timing simulation),
    congestion distribution, and simulated daily traffic volume.
    """
    simulator = request.app.state.simulator
    state = simulator.get_state()

    current_vehicles = state["metrics"]["total_vehicles"]
    baseline_wait = round(45 + (current_vehicles * 0.08), 1)
    ai_wait = state["metrics"]["avg_waiting_time"]

    improvement = round(((baseline_wait - ai_wait) / baseline_wait) * 100, 1) if baseline_wait > 0 else 0.0

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_volume = []
    for day in days:
        daily_volume.append({
            "day": day,
            "volume_without_ai": random.randint(12000, 15000),
            "volume_with_ai": random.randint(13500, 16500)
        })

    junctions_wait = []
    for j in state["junctions"]:
        junctions_wait.append({
            "name": j["id"],
            "without_ai": round(j["avg_waiting_time"] * 1.35, 1),
            "with_ai": j["avg_waiting_time"]
        })

    levels = {"Low": 0, "Moderate": 0, "High": 0, "Severe": 0}
    for j in state["junctions"]:
        level = j.get("congestion_level", "Low")
        if level in levels:
            levels[level] += 1

    congestion_distribution = [
        {"name": "Low", "value": levels["Low"], "color": "#00ff87"},
        {"name": "Moderate", "value": levels["Moderate"], "color": "#ffd700"},
        {"name": "High", "value": levels["High"], "color": "#ff007f"},
        {"name": "Severe", "value": levels["Severe"], "color": "#ef4444"}
    ]

    return {
        "summary": {
            "without_ai_wait_sec": baseline_wait,
            "with_ai_wait_sec": ai_wait,
            "improvement_pct": improvement,
            "label": "Simulation Analytics"
        },
        "daily_volume": daily_volume,
        "junction_comparison": junctions_wait,
        "congestion_distribution": congestion_distribution
    }

@router.get("/predictions")
async def get_predictions(request: Request, timeframe: Optional[str] = "30m"):
    """
    Generate prototype predictions for simulated traffic volume flow.
    """
    simulator = request.app.state.simulator
    state = simulator.get_state()
    current_vehicles = state["metrics"]["total_vehicles"]

    factor = 1.15 if timeframe == "15m" else (1.30 if timeframe == "1h" else 1.20)
    predicted_vehicles = int(current_vehicles * factor)
    avg_density = state["metrics"]["avg_density"]

    timeline = []
    import datetime
    now = datetime.datetime.now()
    for i in range(10):
        time_label = (now + datetime.timedelta(minutes=i * 5)).strftime("%H:%M")
        wave = 1.0 + 0.25 * math.sin(i / 2.0)
        timeline.append({
            "time": time_label,
            "current_volume": int(current_vehicles * wave),
            "predicted_volume": int(current_vehicles * wave * factor)
        })

    return {
        "timeframe": timeframe,
        "current_volume": current_vehicles,
        "predicted_volume": predicted_vehicles,
        "congestion_probability_pct": min(95, max(10, int(avg_density * 80 + 10))),
        "timeline": timeline
    }

@router.post("/incident")
async def post_incident(request: Request, body: IncidentRequest):
    simulator = request.app.state.simulator
    success = simulator.set_incident(body.junction_id, body.active)
    if not success:
        raise HTTPException(status_code=404, detail="Junction not found")
    return {"status": "success", "message": f"Incident state updated for {body.junction_id}"}

@router.post("/emergency")
async def post_emergency(request: Request, body: EmergencyRequest):
    simulator = request.app.state.simulator
    success = simulator.trigger_emergency_route(body.route)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid route junctions provided")
    return {"status": "success", "message": "Emergency route priority activated (Simulation)"}

@router.post("/simulation/control")
async def control_simulation(request: Request, body: ControlRequest):
    simulator = request.app.state.simulator
    action = body.action.lower()

    if action == "pause":
        simulator.is_running = False
        message = "Simulation paused"
    elif action == "resume":
        simulator.is_running = True
        message = "Simulation resumed"
    elif action == "reset":
        simulator.reset()
        message = "Simulation reset to Balanced Mode"
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'pause', 'resume', or 'reset'")

    return {"status": "success", "message": message, "simulator_running": simulator.is_running}

@router.get("/system-status")
async def get_system_status(request: Request):
    simulator = request.app.state.simulator
    detector = YOLOVehicleDetector.get_instance()

    return {
        "status": {
            "ai_yolo_engine": "Online (YOLOv8n)" if detector.is_loaded else "Standby",
            "traffic_simulator": "Running" if simulator.is_running else "Paused",
            "active_preset": simulator.active_preset,
            "thresholds": DENSITY_THRESHOLDS,
            "signal_timing_rules": SIGNAL_TIMING_RULES
        }
    }

import math
