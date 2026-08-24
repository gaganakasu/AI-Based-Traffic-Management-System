import random
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class TrafficSimulator:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.is_running = True
        self.active_preset = "balanced"  # "balanced", "rush", "emergency", "incident"
        self.alerts = []
        self.emergency_route = None
        self.emergency_step = 0
        
        # 6 Prototype Junction Nodes
        self.junctions = {
            "Junction A": {
                "id": "Junction A",
                "code": "J01",
                "short_id": "J1",
                "name": "Junction 01 (Downtown Cross)",
                "lat": 17.4483,
                "lng": 78.3741,
                "directions": {
                    "North-South": {"vehicle_count": 24, "cars": 16, "motorcycles": 5, "buses": 2, "trucks": 1, "queue_length": 6, "density": 0.40, "waiting_time": 14.5, "signal": "Green", "timer": 30},
                    "East-West": {"vehicle_count": 14, "cars": 9, "motorcycles": 3, "buses": 1, "trucks": 1, "queue_length": 3, "density": 0.25, "waiting_time": 8.0, "signal": "Red", "timer": 30}
                },
                "current_active": "North-South",
                "ai_optimized": True,
                "has_incident": False,
                "incident_desc": "",
                "emergency_override": False,
                "congestion_level": "Moderate",
                "base_green_time": 30
            },
            "Junction B": {
                "id": "Junction B",
                "code": "J02",
                "short_id": "J2",
                "name": "Junction 02 (Subway Plaza)",
                "lat": 17.4265,
                "lng": 78.4123,
                "directions": {
                    "North-South": {"vehicle_count": 12, "cars": 8, "motorcycles": 2, "buses": 1, "trucks": 1, "queue_length": 2, "density": 0.20, "waiting_time": 6.5, "signal": "Red", "timer": 25},
                    "East-West": {"vehicle_count": 22, "cars": 14, "motorcycles": 4, "buses": 2, "trucks": 2, "queue_length": 5, "density": 0.35, "waiting_time": 12.0, "signal": "Green", "timer": 25}
                },
                "current_active": "East-West",
                "ai_optimized": True,
                "has_incident": False,
                "incident_desc": "",
                "emergency_override": False,
                "congestion_level": "Low",
                "base_green_time": 30
            },
            "Junction C": {
                "id": "Junction C",
                "code": "J03",
                "short_id": "J3",
                "name": "Junction 03 (Financial District)",
                "lat": 17.4401,
                "lng": 78.3489,
                "directions": {
                    "North-South": {"vehicle_count": 28, "cars": 18, "motorcycles": 6, "buses": 2, "trucks": 2, "queue_length": 8, "density": 0.48, "waiting_time": 18.0, "signal": "Green", "timer": 30},
                    "East-West": {"vehicle_count": 15, "cars": 10, "motorcycles": 3, "buses": 1, "trucks": 1, "queue_length": 3, "density": 0.26, "waiting_time": 9.0, "signal": "Red", "timer": 30}
                },
                "current_active": "North-South",
                "ai_optimized": True,
                "has_incident": False,
                "incident_desc": "",
                "emergency_override": False,
                "congestion_level": "Moderate",
                "base_green_time": 30
            },
            "Junction D": {
                "id": "Junction D",
                "code": "J04",
                "short_id": "J4",
                "name": "Junction 04 (Industrial Arterial)",
                "lat": 17.4375,
                "lng": 78.4482,
                "directions": {
                    "North-South": {"vehicle_count": 8, "cars": 4, "motorcycles": 1, "buses": 1, "trucks": 2, "queue_length": 1, "density": 0.14, "waiting_time": 4.0, "signal": "Red", "timer": 15},
                    "East-West": {"vehicle_count": 10, "cars": 5, "motorcycles": 2, "buses": 1, "trucks": 2, "queue_length": 2, "density": 0.16, "waiting_time": 5.0, "signal": "Green", "timer": 15}
                },
                "current_active": "East-West",
                "ai_optimized": True,
                "has_incident": False,
                "incident_desc": "",
                "emergency_override": False,
                "congestion_level": "Low",
                "base_green_time": 30
            },
            "Junction E": {
                "id": "Junction E",
                "code": "J05",
                "short_id": "J5",
                "name": "Junction 05 (Residential Boulevard)",
                "lat": 17.3616,
                "lng": 78.4747,
                "directions": {
                    "North-South": {"vehicle_count": 6, "cars": 4, "motorcycles": 1, "buses": 1, "trucks": 0, "queue_length": 1, "density": 0.10, "waiting_time": 3.0, "signal": "Green", "timer": 18},
                    "East-West": {"vehicle_count": 5, "cars": 3, "motorcycles": 1, "buses": 1, "trucks": 0, "queue_length": 1, "density": 0.08, "waiting_time": 2.5, "signal": "Red", "timer": 18}
                },
                "current_active": "North-South",
                "ai_optimized": True,
                "has_incident": False,
                "incident_desc": "",
                "emergency_override": False,
                "congestion_level": "Low",
                "base_green_time": 30
            },
            "Junction F": {
                "id": "Junction F",
                "code": "J06",
                "short_id": "J6",
                "name": "Junction 06 (Highway Gateway)",
                "lat": 17.4435,
                "lng": 78.4983,
                "directions": {
                    "North-South": {"vehicle_count": 18, "cars": 12, "motorcycles": 3, "buses": 1, "trucks": 2, "queue_length": 4, "density": 0.30, "waiting_time": 10.0, "signal": "Red", "timer": 20},
                    "East-West": {"vehicle_count": 20, "cars": 13, "motorcycles": 4, "buses": 1, "trucks": 2, "queue_length": 5, "density": 0.32, "waiting_time": 11.0, "signal": "Green", "timer": 20}
                },
                "current_active": "East-West",
                "ai_optimized": True,
                "has_incident": False,
                "incident_desc": "",
                "emergency_override": False,
                "congestion_level": "Low",
                "base_green_time": 30
            }
        }
        self.add_alert("System Ready", "AI Traffic Simulation initialized with adaptive signal control logic.", "info")

    def calculate_adaptive_green_time(self, vehicle_count: int) -> int:
        """
        Adaptive traffic signal optimization rules (Prototype demo standards):
        - Low (0–5 vehicles): 15 seconds
        - Medium (6–15 vehicles): 30 seconds
        - High (>15 vehicles): 45 seconds
        """
        if vehicle_count <= 5:
            return 15
        elif vehicle_count <= 15:
            return 30
        else:
            return 45

    def apply_preset(self, preset: str) -> Dict[str, Any]:
        """
        Applies functional scenario presets:
        1. 'balanced': Normal conditions (Adaptive / Balanced)
        2. 'rush': High volume (Peak-hour congestion simulation, extended green 45s)
        3. 'emergency': Clearly labelled simulation prioritizing emergency direction (North/South)
        4. 'incident': Clearly labelled simulated incident affecting direction (East/West)
        """
        self.active_preset = preset.lower()

        if self.active_preset == "balanced":
            self.clear_emergency_route()
            for j_id, j in self.junctions.items():
                j["has_incident"] = False
                j["incident_desc"] = ""
                j["emergency_override"] = False
                j["directions"]["North-South"]["vehicle_count"] = random.randint(8, 14)
                j["directions"]["East-West"]["vehicle_count"] = random.randint(6, 12)
                for d in ["North-South", "East-West"]:
                    vc = j["directions"][d]["vehicle_count"]
                    j["directions"][d]["cars"] = int(vc * 0.65)
                    j["directions"][d]["motorcycles"] = int(vc * 0.15)
                    j["directions"][d]["buses"] = max(1, int(vc * 0.10))
                    j["directions"][d]["trucks"] = max(0, vc - j["directions"][d]["cars"] - j["directions"][d]["motorcycles"] - j["directions"][d]["buses"])
                    j["directions"][d]["density"] = round(vc / 30.0, 2)
                    j["directions"][d]["queue_length"] = max(1, int(vc * 0.25))
                    j["directions"][d]["waiting_time"] = round(j["directions"][d]["queue_length"] * 1.5, 1)
                j["congestion_level"] = "Low" if max(j["directions"]["North-South"]["density"], j["directions"]["East-West"]["density"]) < 0.4 else "Moderate"

            self.add_alert(
                "Preset: Balanced Mode",
                "Normal simulated traffic conditions calibrated. Signals operating on standard adaptive cycles.",
                "success"
            )

        elif self.active_preset == "rush":
            self.clear_emergency_route()
            for j_id, j in self.junctions.items():
                j["has_incident"] = False
                j["incident_desc"] = ""
                j["emergency_override"] = False
                # Significant traffic volume on North-South corridor
                j["directions"]["North-South"]["vehicle_count"] = random.randint(24, 35)
                j["directions"]["East-West"]["vehicle_count"] = random.randint(12, 18)
                for d in ["North-South", "East-West"]:
                    vc = j["directions"][d]["vehicle_count"]
                    j["directions"][d]["cars"] = int(vc * 0.70)
                    j["directions"][d]["motorcycles"] = int(vc * 0.15)
                    j["directions"][d]["buses"] = max(1, int(vc * 0.10))
                    j["directions"][d]["trucks"] = max(1, vc - j["directions"][d]["cars"] - j["directions"][d]["motorcycles"] - j["directions"][d]["buses"])
                    j["directions"][d]["density"] = min(1.0, round(vc / 30.0, 2))
                    j["directions"][d]["queue_length"] = int(vc * 0.45)
                    j["directions"][d]["waiting_time"] = round(j["directions"][d]["queue_length"] * 1.8, 1)
                j["congestion_level"] = "High"

            self.add_alert(
                "Preset: Rush Hour Mode",
                "Peak-hour congestion detected (Simulation). Adaptive system recommending extended green phases (45s).",
                "warning"
            )

        elif self.active_preset == "emergency":
            self.clear_emergency_route()
            self.trigger_emergency_route(["Junction A", "Junction B", "Junction D"])
            self.add_alert(
                "Preset: Emergency Priority Mode",
                "Simulated emergency priority gives temporary signal preference to North/South direction.",
                "emergency"
            )

        elif self.active_preset == "incident":
            self.clear_emergency_route()
            # Simulate accident at Junction A (East-West corridor)
            j = self.junctions["Junction A"]
            j["has_incident"] = True
            j["incident_desc"] = "Simulated road obstruction on East-West approach."
            j["directions"]["East-West"]["vehicle_count"] = 32
            j["directions"]["East-West"]["queue_length"] = 18
            j["directions"]["East-West"]["density"] = 0.85
            j["directions"]["East-West"]["waiting_time"] = 35.0
            j["congestion_level"] = "High"

            self.add_alert(
                "Preset: Incident Mode",
                "Simulated incident active at Junction A (East-West). System applying adaptive signal compensation.",
                "danger"
            )

        return self.get_state()

    def tick(self, is_peak_hour: int = 0) -> Dict[str, Any]:
        """
        Simulation tick executing every second.
        Updates light timers, cycle transitions, and vehicle queues.
        """
        if not self.is_running:
            return self.get_state()

        self.update_emergency_sequence()

        for j_id, j_data in self.junctions.items():
            active = j_data["current_active"]
            inactive = "East-West" if active == "North-South" else "North-South"

            # 1. Decrement timers
            if j_data["directions"][active]["timer"] > 0:
                j_data["directions"][active]["timer"] -= 1
                j_data["directions"][inactive]["timer"] -= 1
            else:
                current_signal = j_data["directions"][active]["signal"]
                if current_signal == "Green":
                    j_data["directions"][active]["signal"] = "Yellow"
                    j_data["directions"][active]["timer"] = 3
                    j_data["directions"][inactive]["timer"] = 3
                elif current_signal == "Yellow":
                    # Transition to next direction
                    j_data["current_active"] = inactive
                    j_data["directions"][inactive]["signal"] = "Green"
                    j_data["directions"][active]["signal"] = "Red"

                    # Calculate adaptive green timer based on vehicle load
                    next_vc = j_data["directions"][inactive]["vehicle_count"]
                    new_timer = self.calculate_adaptive_green_time(next_vc)

                    # Handle emergency override lock
                    if j_data["emergency_override"]:
                        override_dir = j_data.get("emergency_priority_dir", inactive)
                        if override_dir == inactive:
                            new_timer = 45
                            j_data["directions"][inactive]["signal"] = "Green"
                            j_data["directions"][active]["signal"] = "Red"
                        else:
                            j_data["current_active"] = override_dir
                            j_data["directions"][override_dir]["signal"] = "Green"
                            j_data["directions"][active if override_dir == inactive else inactive]["signal"] = "Red"
                            new_timer = 45

                    j_data["directions"][inactive]["timer"] = new_timer
                    j_data["directions"][active]["timer"] = new_timer

            # 2. Dynamic vehicle inflows/outflows
            for direction in ["North-South", "East-West"]:
                st = j_data["directions"][direction]
                if random.random() < 0.25:
                    inflow = 1 if not j_data["has_incident"] else random.randint(1, 2)
                    if self.active_preset == "rush" and direction == "North-South":
                        inflow += 1
                    st["vehicle_count"] = min(60, st["vehicle_count"] + inflow)

                # Departures on green
                if st["signal"] == "Green" and random.random() < 0.40:
                    st["vehicle_count"] = max(2, st["vehicle_count"] - random.randint(1, 2))

                # Update breakdown & queue
                vc = st["vehicle_count"]
                st["cars"] = max(1, int(vc * 0.65))
                st["motorcycles"] = max(0, int(vc * 0.15))
                st["buses"] = max(0, int(vc * 0.10))
                st["trucks"] = max(0, vc - st["cars"] - st["motorcycles"] - st["buses"])
                st["density"] = min(1.0, round(vc / 30.0, 2))
                st["queue_length"] = max(0, int(vc * (0.45 if st["signal"] == "Red" else 0.20)))
                st["waiting_time"] = round(st["queue_length"] * 1.5, 1)

            # 3. Congestion classification
            max_d = max(j_data["directions"]["North-South"]["density"], j_data["directions"]["East-West"]["density"])
            if max_d < 0.35:
                j_data["congestion_level"] = "Low"
            elif max_d < 0.70:
                j_data["congestion_level"] = "Moderate"
            else:
                j_data["congestion_level"] = "High"

        return self.get_state()

    def find_junction_key(self, query: str) -> Optional[str]:
        if not query:
            return None
        q = str(query).strip().lower()
        for k, v in self.junctions.items():
            if (k.lower() == q or 
                v.get("id", "").lower() == q or 
                v.get("code", "").lower() == q or 
                v.get("short_id", "").lower() == q or 
                v.get("name", "").lower() == q or
                q in v.get("name", "").lower()):
                return k
        return None

    def set_incident(self, junction_id: str, active: bool) -> bool:
        real_key = self.find_junction_key(junction_id)
        if not real_key:
            return False
        j = self.junctions[real_key]
        j["has_incident"] = active
        if active:
            j["incident_desc"] = "Simulated road obstruction on East-West approach."
            j["directions"]["East-West"]["vehicle_count"] += 15
            j["congestion_level"] = "High"
            self.add_alert("Incident Reported (Simulated)", f"Roadway obstruction simulated at {j['name']}.", "danger")
        else:
            j["incident_desc"] = ""
            j["directions"]["East-West"]["vehicle_count"] = max(5, j["directions"]["East-West"]["vehicle_count"] - 12)
            self.add_alert("Incident Cleared (Simulated)", f"Roadway cleared at {j['name']}.", "success")
        return True

    def trigger_emergency_route(self, route: List[str]) -> bool:
        self.clear_emergency_route()
        valid = []
        for r in route:
            k = self.find_junction_key(r)
            if k and k not in valid:
                valid.append(k)
        if not valid:
            return False
        self.emergency_route = valid
        self.emergency_step = 0
        self.apply_emergency_override_step()
        return True

    def update_emergency_sequence(self):
        if not self.emergency_route:
            return
        current_j_id = self.emergency_route[self.emergency_step]
        j = self.junctions[current_j_id]
        j["emergency_override"] = True

        if random.random() < 0.15:  # progress step
            j["emergency_override"] = False
            self.emergency_step += 1
            if self.emergency_step >= len(self.emergency_route):
                self.add_alert(
                    "Emergency Route Cleared (Simulated)",
                    "Simulated priority corridor passed. Standard adaptive timings restored.",
                    "success"
                )
                self.clear_emergency_route()
            else:
                self.apply_emergency_override_step()

    def apply_emergency_override_step(self):
        if not self.emergency_route or self.emergency_step >= len(self.emergency_route):
            return
        j_id = self.emergency_route[self.emergency_step]
        j = self.junctions[j_id]
        j["emergency_override"] = True
        override_dir = "North-South"
        j["emergency_priority_dir"] = override_dir
        j["directions"][override_dir]["signal"] = "Green"
        j["directions"][override_dir]["timer"] = 45
        other_dir = "East-West"
        j["directions"][other_dir]["signal"] = "Red"
        j["directions"][other_dir]["timer"] = 45
        j["current_active"] = override_dir

    def clear_emergency_route(self):
        if self.emergency_route:
            for j_id in self.emergency_route:
                if j_id in self.junctions:
                    self.junctions[j_id]["emergency_override"] = False
                    self.junctions[j_id]["emergency_priority_dir"] = None
        self.emergency_route = None
        self.emergency_step = 0

    def add_alert(self, title: str, message: str, type_: str = "info"):
        alert = {
            "id": random.randint(1000, 9999),
            "title": title,
            "message": message,
            "type": type_,
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
        }
        self.alerts.insert(0, alert)
        self.alerts = self.alerts[:10]

    def get_state(self) -> Dict[str, Any]:
        total_vehicles = 0
        total_cars = 0
        total_motos = 0
        total_buses = 0
        total_trucks = 0
        congested_count = 0
        total_waiting_time = 0
        active_junctions = len(self.junctions)
        total_density = 0

        junctions_list = []
        for j_id, j_data in self.junctions.items():
            j_veh = sum(j_data["directions"][d]["vehicle_count"] for d in ["North-South", "East-West"])
            j_cars = sum(j_data["directions"][d].get("cars", 0) for d in ["North-South", "East-West"])
            j_motos = sum(j_data["directions"][d].get("motorcycles", 0) for d in ["North-South", "East-West"])
            j_buses = sum(j_data["directions"][d].get("buses", 0) for d in ["North-South", "East-West"])
            j_trucks = sum(j_data["directions"][d].get("trucks", 0) for d in ["North-South", "East-West"])

            j_waiting = sum(j_data["directions"][d]["waiting_time"] for d in ["North-South", "East-West"]) / 2
            j_density = sum(j_data["directions"][d]["density"] for d in ["North-South", "East-West"]) / 2

            total_vehicles += j_veh
            total_cars += j_cars
            total_motos += j_motos
            total_buses += j_buses
            total_trucks += j_trucks
            total_waiting_time += j_waiting
            total_density += j_density

            if j_data["congestion_level"] in ["High", "Severe"]:
                congested_count += 1

            junctions_list.append({
                "id": j_id,
                **j_data,
                "total_vehicles": j_veh,
                "avg_waiting_time": round(j_waiting, 1),
                "avg_density": round(j_density, 2)
            })

        avg_waiting = round(total_waiting_time / active_junctions, 1)
        avg_density = round(total_density / active_junctions, 2)

        # Scenario descriptor details
        scenario_info = {
            "preset": self.active_preset,
            "label": {
                "balanced": "Balanced Mode",
                "rush": "Rush Hour Mode",
                "emergency": "Emergency Priority Mode (Simulation)",
                "incident": "Incident Mode (Simulation)"
            }.get(self.active_preset, "Balanced Mode"),
            "traffic_condition": "HIGH" if (self.active_preset in ["rush", "incident"] or congested_count > 0) else "NORMAL",
            "signal_strategy": {
                "balanced": "Adaptive / Balanced",
                "rush": "Extended Green / Congestion Reduction",
                "emergency": "Emergency Priority Green Wave",
                "incident": "Adaptive Signal Compensation"
            }.get(self.active_preset, "Adaptive / Balanced"),
            "status_text": {
                "balanced": "Normal simulated traffic flow. Signals operating on standard adaptive cycles.",
                "rush": "Peak-hour congestion detected (simulation). Extended green light timers active.",
                "emergency": "Simulated emergency priority gives temporary signal preference to North/South direction.",
                "incident": "Simulated traffic incident active. System adjusting signal timings dynamically."
            }.get(self.active_preset, "Normal traffic conditions.")
        }

        # Calculate primary focus junction (Junction A) metrics
        prim_j = self.junctions["Junction A"]
        prim_active = prim_j["current_active"]
        prim_inactive = "East-West" if prim_active == "North-South" else "North-South"
        current_signal_label = f"🟢 {prim_active.upper()}" if prim_j["directions"][prim_active]["signal"] == "Green" else f"🟡 {prim_active.upper()}"
        next_signal_label = prim_inactive.upper()
        time_remaining = prim_j["directions"][prim_active]["timer"]
        recommended_green = self.calculate_adaptive_green_time(prim_j["directions"][prim_active]["vehicle_count"])

        # Dynamic explanation for AI Signal Recommendation Card
        if prim_j["directions"]["North-South"]["vehicle_count"] >= prim_j["directions"]["East-West"]["vehicle_count"]:
            ai_recommendation_text = f"North/South approach currently has the higher traffic density ({prim_j['directions']['North-South']['vehicle_count']} vehicles). The system recommends a green phase duration of {self.calculate_adaptive_green_time(prim_j['directions']['North-South']['vehicle_count'])} seconds."
        else:
            ai_recommendation_text = f"East/West approach currently has the higher traffic density ({prim_j['directions']['East-West']['vehicle_count']} vehicles). The system recommends a green phase duration of {self.calculate_adaptive_green_time(prim_j['directions']['East-West']['vehicle_count'])} seconds."

        return {
            "metrics": {
                "total_vehicles": total_vehicles,
                "cars": total_cars,
                "motorcycles": total_motos,
                "buses": total_buses,
                "trucks": total_trucks,
                "active_intersections": active_junctions,
                "congested_roads": congested_count,
                "avg_waiting_time": avg_waiting,
                "avg_density": avg_density,
                "ai_status": "ONLINE (Adaptive Rules)",
                "emergency_mode": "ACTIVE (SIMULATION)" if self.emergency_route else "STANDBY",
                "current_signal": current_signal_label,
                "next_signal": next_signal_label,
                "time_remaining_sec": time_remaining,
                "recommended_green_sec": recommended_green,
                "ai_recommendation_explanation": ai_recommendation_text
            },
            "scenario": scenario_info,
            "junctions": junctions_list,
            "alerts": self.alerts,
            "emergency_route": self.emergency_route,
            "emergency_step": self.emergency_step
        }

    def reset(self) -> Dict[str, Any]:
        return self.apply_preset("balanced")
