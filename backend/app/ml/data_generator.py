import csv
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filename="traffic_training_data.csv", num_records=1500):
    """
    Generates a realistic traffic dataset for training.
    Saved to the specified filename.
    """
    intersections = ["Junction A", "Junction B", "Junction C", "Junction D", "Junction E", "Junction F"]
    weather_options = ["Sunny", "Rainy", "Cloudy", "Foggy"]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp", "intersection_id", "vehicle_count", "traffic_density", 
            "queue_length", "average_speed", "waiting_time", "time_of_day", 
            "weather", "is_peak_hour", "optimal_green_time"
        ])
        
        start_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_records):
            timestamp = start_time + timedelta(minutes=i * 10)
            intersection = random.choice(intersections)
            
            # Determine peak hour based on hour of day
            hour = timestamp.hour
            is_peak = 1 if ((8 <= hour <= 10) or (17 <= hour <= 19)) else 0
            
            # Weather impacts speeds and queues
            weather = random.choices(weather_options, weights=[0.6, 0.2, 0.15, 0.05])[0]
            
            # Determine base counts based on intersection type and peak hours
            if intersection in ["Junction A", "Junction F"]:  # Busy downtown / highway
                base_vehicles = random.randint(80, 150) if is_peak else random.randint(30, 80)
            elif intersection in ["Junction C", "Junction D"]:  # Commercial / industrial
                base_vehicles = random.randint(60, 120) if is_peak else random.randint(25, 60)
            else:  # Residential / suburban
                base_vehicles = random.randint(40, 85) if is_peak else random.randint(10, 45)
                
            # Add some weather penalty to traffic
            if weather == "Rainy":
                base_vehicles = int(base_vehicles * 1.15)
            elif weather == "Foggy":
                base_vehicles = int(base_vehicles * 1.05)
                
            vehicle_count = max(5, base_vehicles + random.randint(-15, 15))
            
            # Traffic density (0.0 to 1.0)
            max_capacity = 200
            traffic_density = min(1.0, vehicle_count / max_capacity)
            
            # Queue length
            queue_length = int(vehicle_count * random.uniform(0.2, 0.45))
            if weather == "Rainy":
                queue_length = int(queue_length * 1.2)
            
            # Average speed (km/h) - decreases as density increases
            base_speed = 60 if intersection == "Junction F" else 40
            average_speed = max(5.0, base_speed - (traffic_density * 30.0) - (random.uniform(0.0, 5.0)))
            if weather in ["Rainy", "Foggy"]:
                average_speed *= 0.85
                
            # Average waiting time (seconds)
            waiting_time = max(5.0, (queue_length * 1.5) + (traffic_density * 15.0) + random.uniform(-3, 3))
            
            # Formulate the TARGET variable: optimal_green_time (seconds)
            # A good system gives more green time to dense intersections
            # Base green time is 30 seconds
            # Range: 15s to 90s
            green_time_calc = 25.0
            green_time_calc += (traffic_density * 45.0)  # Max +45s from density
            green_time_calc += (queue_length * 0.5)      # Max +~25s from queue
            if is_peak:
                green_time_calc += 10.0                   # Peak hour bonus
            
            optimal_green_time = int(min(90, max(15, green_time_calc + random.uniform(-3, 3))))
            
            # format time of day (HH:MM)
            time_of_day = timestamp.strftime("%H:%M")
            
            writer.writerow([
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                intersection,
                vehicle_count,
                round(traffic_density, 3),
                queue_length,
                round(average_speed, 1),
                round(waiting_time, 1),
                time_of_day,
                weather,
                is_peak,
                optimal_green_time
            ])
            
    print(f"Synthetic dataset of {num_records} rows successfully created at {filename}")

if __name__ == "__main__":
    # Generate data in standard location
    generate_synthetic_data("../../data/traffic_training_data.csv")
