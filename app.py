import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import cv2
import sys
import logging

# Set logging level to WARNING or ERROR to suppress lower level logs
logging.basicConfig(level=logging.WARNING)

transform = transforms.Compose([
    transforms.ToTensor()
])
model_dir = "meter-reading-model-last.pt" if sys.argv[2] == "last" else "meter-reading-model-best.pt"
model = YOLO("meter-reading-model-last.pt") # Load the best model

def predict(image_dir):
    image = plt.imread(image_dir)
    img_tensor = transform(image)

    result = model.predict(img_tensor.unsqueeze(0), ) # Predict the image

    ## classes to actaul number convertion
    s_result = 0
    classes = result[0].boxes.cls
    xy = [int(i[0]) for i in result[0].boxes.xyxy]
    a = sorted(range(len(xy)), key=lambda k: xy[k])
    for idx, i in enumerate(reversed(a)):
        power = 10**idx 
        s_result += int(classes[i])*power
        # print(s_result)
    s_result = s_result/10 # final result

    return {"prediction": s_result}
result = predict(sys.argv[1])
print(result)