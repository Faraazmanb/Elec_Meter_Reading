import os
from ultralytics import YOLO
import logging

# Set logging level to WARNING or ERROR to suppress lower-level logs
logging.basicConfig(level=logging.WARNING)

# Load the YOLO model (default to "best" model)
# script_dir = os.path.dirname(__file__)
# model_path_best = os.path.join(script_dir, "meter-reading-model-best.pt")
# model_path_last = os.path.join(script_dir, "meter-reading-model-last.pt")


script_dir = os.path.dirname(__file__)  # Get the directory of the current script

# Define paths to models inside the 'trainedmodel' folder
model_path_best = os.path.join(script_dir, "trainedmodel", "meter-reading-model-best.pt")
model_path_last = os.path.join(script_dir, "trainedmodel", "meter-reading-model-last.pt")




default_model = YOLO(model_path_best)  # Default to "best" model
print(f"Default model loaded from: {model_path_best}")

def predict(image_path, model_version="best"):
    """
    Perform prediction on an image using the YOLO model.
    
    Args:
        image_path (str): Path to the image file.
        model_version (str): Specify the model version ("best" or "last"). Defaults to "best".
    
    Returns:
        dict: A dictionary containing the prediction result or an error message.
    """
    try:
        # Select the appropriate model
        model_dir = model_path_last if model_version == "last" else model_path_best
        model = YOLO(model_dir)
        print(f"Using model from: {model_dir}")
        
        # Perform prediction directly on the image
        print(f"Reading image from: {image_path}")
        result = model.predict(image_path)

        # Classes to actual number conversion
        s_result = 0
        classes = result[0].boxes.cls
        xy = [int(i[0]) for i in result[0].boxes.xyxy]
        a = sorted(range(len(xy)), key=lambda k: xy[k])
        for idx, i in enumerate(reversed(a)):
            power = 10 ** idx
            s_result += int(classes[i]) * power
        s_result = s_result / 10  # Final result
        print(f"Prediction completed: {s_result}")

        return {"prediction": s_result}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}