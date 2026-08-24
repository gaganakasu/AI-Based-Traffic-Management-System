import asyncio
import os
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.simulation.simulator import TrafficSimulator
from app.websocket.manager import ConnectionManager
from app.api.routes import router as api_router
from app.ml.yolo_detector import YOLOVehicleDetector

# Initialize components
manager = ConnectionManager()
simulator = TrafficSimulator()

async def simulation_loop(app: FastAPI):
    """
    Background simulation loop running every second.
    Computes traffic updates, applies adaptive signal timing,
    and broadcasts state via WebSocket.
    """
    print("[SimulationLoop] Started background traffic loop.")
    while True:
        try:
            if app.state.simulator.is_running:
                import datetime
                hour = datetime.datetime.now().hour
                is_peak = 1 if ((8 <= hour <= 10) or (17 <= hour <= 19)) else 0

                state = app.state.simulator.tick(is_peak_hour=is_peak)
                await manager.broadcast(state)

            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            print("[SimulationLoop] Simulation loop cancelled.")
            break
        except Exception as e:
            print(f"[SimulationLoop] Exception in loop: {e}")
            await asyncio.sleep(1.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize YOLO detector and ensure demo video is available
    base_dir = os.path.abspath(os.path.dirname(__file__))
    sample_video_path = os.path.join(base_dir, "data", "sample_traffic.mp4")

    # Pre-initialize YOLO model instance
    try:
        detector = YOLOVehicleDetector.get_instance()
        if not os.path.exists(sample_video_path):
            print("[Startup] Generating sample demo traffic video...")
            YOLOVehicleDetector.generate_demo_traffic_video(sample_video_path, duration_sec=4, fps=25)
    except Exception as e:
        print(f"[Startup] Note on YOLO initialization: {e}")

    # Launch simulation loop in background
    sim_task = asyncio.create_task(simulation_loop(app))

    yield
    # Shutdown
    sim_task.cancel()
    try:
        await sim_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="AI-Based Traffic Management System",
    description="Prototype AI/Computer Vision Traffic Management & Adaptive Signal Optimization Dashboard",
    version="2.0.0",
    lifespan=lifespan
)

app.state.simulator = simulator

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST endpoints
app.include_router(api_router, prefix="/api")

from fastapi.responses import HTMLResponse

@app.get("/")
async def get_index():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    index_path = os.path.join(base_dir, "..", "frontend", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>AI-Based Traffic Management System: Frontend file not found</h1>", status_code=404)

@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        current_state = app.state.simulator.get_state()
        await websocket.send_json(current_state)

        while True:
            data = await websocket.receive_text()
            try:
                event = json.loads(data)
                if "preset" in event:
                    app.state.simulator.apply_preset(event["preset"])
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
