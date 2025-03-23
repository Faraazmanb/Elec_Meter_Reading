import os
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.WARNING)

# Set model paths
script_dir = os.path.dirname(__file__)
model_path_best = os.path.join(script_dir, "trainedmodel", "meter-reading-model-best.pt")
model_path_last = os.path.join(script_dir, "trainedmodel", "meter-reading-model-last.pt")

# Load default model
default_model = YOLO(model_path_best)
print(f"Default model loaded from: {model_path_best}")

# Initialize SQLite database
db_path = os.path.join(script_dir, "meter_readings.db")

def init_db():
    """Initialize the database and create the table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                reading REAL
            )
        ''')
        conn.commit()

def get_previous_reading():
    """Retrieve the most recent reading from the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT reading FROM readings ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else None

def save_reading(reading):
    """Save the extracted meter reading to the database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO readings (timestamp, reading) VALUES (?, ?)", (timestamp, reading))
        conn.commit()

def predict(image_path, model_version="best"):
    """Perform prediction, store in DB, and calculate consumption."""
    try:
        model_dir = model_path_last if model_version == "last" else model_path_best
        model = YOLO(model_dir)
        print(f"Using model from: {model_dir}")

        # Perform prediction
        print(f"Reading image from: {image_path}")
        results = model.predict(image_path)

        # Extract reading and accuracy
        s_result = 0
        classes = results[0].boxes.cls
        xy = [int(i[0]) for i in results[0].boxes.xyxy]
        a = sorted(range(len(xy)), key=lambda k: xy[k])
        
        for idx, i in enumerate(reversed(a)):
            power = 10 ** idx
            s_result += int(classes[i]) * power
        s_result = s_result / 10  # Final result
        
        # Extract accuracy (confidence score)
        confidences = results[0].boxes.conf  # Confidence scores of detections
        accuracy = round(float(confidences.mean()) * 100, 2) if len(confidences) > 0 else None

        # Get previous reading and calculate consumption
        previous_reading = get_previous_reading()
        consumption = s_result - previous_reading if previous_reading is not None else None

        # Save the new reading
        save_reading(s_result)

        return {
            "prediction": s_result,
            "accuracy": accuracy,  # Include accuracy in response
            "previous_reading": previous_reading,
            "consumption": consumption
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

# Initialize the database
init_db()
