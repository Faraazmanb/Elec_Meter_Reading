import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import logging
import os
from flask import Flask,render_template,request,send_from_directory

# Set logging level to WARNING or ERROR to suppress lower-level logs
logging.basicConfig(level=logging.WARNING)

model_path_best ="C:\\Users\\mdfar\\Downloads\\finalproj\\finalYearModel\\trainedmodel\\meter-reading-model-best.pt"
model_path_last = "C:\\Users\\mdfar\\Downloads\\finalproj\\finalYearModel\\trainedmodel\\meter-reading-model-last.pt"
default_model = YOLO(model_path_best)  
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
        # print(f"Using model from: {model_dir}")
        print("Using CNN model digit_recognition_model.keras")
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


# Example usage for cnn_model.py:
if __name__ == "__main__":
    # Example image path (you can pass any image path here)
    example_image_path = "C:\\Users\\mdfar\\Downloads\\finalproj\\finalYearModel\\testdata\\images\\30_JPG.rf.56c6c7a52dd55089bb5511a4893a206c.jpg"
    if not os.path.exists(example_image_path):
        print(f"Image file not found: {example_image_path}")





    else:
        result = predict(example_image_path, model_version="best")
        print("Final Result:", result)