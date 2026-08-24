# AI-Based Traffic Management System

**Educational & Capstone Prototype:** Intelligent Traffic Control Room Dashboard, Computer Vision Vehicle Detection, Traffic Density Analysis & Adaptive Signal Timing Recommendation.

---

## 1. Project Title & Purpose

### **AI-Based Traffic Management System**
The application is a prototype traffic-management/control dashboard that uses AI/computer vision to analyze traffic information, estimate congestion, and recommend adaptive traffic-signal timings.

- **Primary Target Users:** Traffic-management operators and control-room personnel.
- **Beneficiaries:** Commuters and urban citizens (indirect beneficiaries).
- **Scope Notice:** This is an educational/capstone prototype and simulation. It demonstrates how adaptive traffic signal strategies can be recommended based on video and sensor inputs, and does not directly control real-world physical traffic signals.

---

## 2. Core Architecture & Workflow

The system is built around a real-time smart traffic-control room simulation:

```
      SIMULATED LIVE TRAFFIC DATA
                  ↓
       CITY TRAFFIC NETWORK (6 Nodes)
                  ↓
       VEHICLE / DENSITY METRICS
                  ↓
        DENSITY THRESHOLD ANALYSIS
                  ↓
       ADAPTIVE SIGNAL TIMING LOGIC (15s/30s/45s)
                  ↓
       CONTROL ROOM DASHBOARD & MAP
```

Optional Experimental Vision Module:
```
TRAFFIC CAMERA / VIDEO FOOTAGE → YOLOv8 DETECTION → CLASSIFICATION → DENSITY → ADAPTIVE TIMING
```

---

## 3. Prototype Signal Optimization Logic

The adaptive signal logic dynamically adjusts green-light phase duration based on the detected or simulated vehicle volume:

| Traffic Density | Vehicle Count Threshold | Recommended Green Phase |
|---|---|---|
| 🟢 **LOW** | 0 – 5 vehicles | **15 seconds** |
| 🟡 **MEDIUM** | 6 – 15 vehicles | **30 seconds** |
| 🔴 **HIGH** | &gt; 15 vehicles | **45 seconds** |

*Note: Thresholds and timings are transparent prototype demonstration rules.*

---

## 4. Functional Scenario Presets

The dashboard top bar provides four mutually exclusive scenario presets:

1. **🟢 Balanced Mode:**
   - Represents normal traffic flow conditions with balanced vehicle distribution across approaches (8–14 vehicles).
   - Signal strategy: *Adaptive / Balanced*.
2. **🚗 Rush Hour Mode:**
   - Represents peak-hour traffic conditions with elevated vehicle volume (24–35 vehicles/approach).
   - Signal strategy: *Extended Green / Congestion Reduction (45s)*.
3. **🚨 Emergency Mode (Simulation):**
   - Simulated emergency vehicle priority giving temporary preference to the selected corridor (J02 ➔ J01 ➔ J04).
   - Signal strategy: *Emergency Priority Green Wave*.
4. **⚠️ Incident Mode (Simulation):**
   - Simulated traffic incident/obstruction at Junction 01 (East-West approach).
   - Signal strategy: *Adaptive Signal Adjustment & Compensation*.

---

## 5. Dashboard Features & Structure

- **Overview Control:** High-level operational metrics (Live Vehicles with category breakdown, Traffic Density level, Congested Junctions count, Active Signals count, AI Signal Recommendation card, and 6-node Network status strip).
- **Live Traffic Map:** Full-screen interactive city network topology view & Leaflet GIS map with 6 interconnected intersections (J01 to J06), real-time density badges (Low/Medium/High), signal states with countdowns, and highlighted emergency corridors.
- **4-Way Junction Visualizer:** Visual 4-way intersection (North, South, East, West) with LED traffic light heads, live countdown timers, approach telemetry, and animated vehicle flow.
- **AI Signal Optimization:** Live network decision matrix for all 6 junctions, density thresholds, and adaptive decision rationale.
- **Scenario Simulation Sandbox:** Interactive sliders to test how varying vehicle volumes trigger different signal recommendations.
- **Analytics Center:** Simulated daily volume, average delay comparisons, and congestion distribution.
- **System Controls:** Simulation engine state (Speed 1x/2x/4x, Pause, Resume, Reset) and optional YOLOv8 Camera Vision demo.

---

## 6. Project Architecture

```
AI-TMS-ANTI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py           # REST endpoints (/preset, /video/sample, /video/analyze)
│   │   ├── ml/
│   │   │   └── yolo_detector.py    # YOLOv8 vehicle detection & classification
│   │   ├── simulation/
│   │   │   └── simulator.py        # 4-way traffic simulation engine & scenario presets
│   │   └── websocket/
│   │       └── manager.py          # WebSocket broadcast manager
│   ├── data/
│   │   └── sample_traffic.mp4      # Demo traffic video for CV testing
│   └── main.py                     # FastAPI application entry point
├── frontend/
│   └── index.html                  # Control-room dashboard UI
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 7. How to Run the Application

### **Prerequisites**
- Python 3.9+

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Start the Backend Server**
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Step 3: Open the Dashboard**
Open your browser and navigate to:
```
http://localhost:8000
```
*(Or open `frontend/index.html` directly in your browser for local standalone simulation mode).*

---

## 8. Limitations & Scope

- **Prototype & Simulation:** This system is an academic capstone prototype designed to demonstrate computer vision vehicle detection and adaptive signal logic. It is not connected to physical municipal traffic hardware.
- **Pretrained Weights:** Computer vision vehicle detection uses lightweight pretrained YOLOv8 (`yolov8n.pt`) without custom domain fine-tuning.
