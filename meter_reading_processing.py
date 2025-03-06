import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Function to train and save the digit recognition model
def train_and_save_model():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the images
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

    # Build the model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Save the model with the recommended format
    model.save('digit_recognition_model.keras')
    print("Model trained and saved as 'digit_recognition_model.keras'")

# Function to load the model
def load_model():
    return tf.keras.models.load_model('digit_recognition_model.keras')

# Function to predict the digit from an image
# Function to predict the digit from an image
def predict_digit(image):
    model = load_model()
    
    # Check if the image is grayscale (1 channel) or not
    if len(image.shape) == 3:  # Image has 3 channels (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    elif len(image.shape) == 2:  # Image is already grayscale
        pass  # No need to convert, it's already grayscale

    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (28, 28))  # Resize to match MNIST input size
    image = image.astype('float32') / 255  # Normalize the image
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model input
    prediction = model.predict(image)
    return np.argmax(prediction)


# Function to preprocess the image
def preprocess_image(img_path, path, debug=True):
    print(f"Processing {img_path}")

    # Read image
    imgArr_o = cv2.imread(path + img_path)

    # Convert to HSV (Hue, Saturation, Value) color space
    imgArr = cv2.cvtColor(imgArr_o, cv2.COLOR_BGR2HSV)

    # Define the range for ROI (Region of Interest)
    roi_lower = np.array([40, 25, 0])
    roi_upper = np.array([80, 255, 255])

    # Create a mask using the defined range
    mask = cv2.inRange(imgArr, roi_lower, roi_upper)

    # Bitwise-AND mask and original image to isolate the region of interest
    imgArr = cv2.bitwise_and(imgArr_o, imgArr_o, mask=mask)

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Add buffers around the detected contour for extraction
        wbuffer = 0.75 * w
        hbuffer = 0.1 * h
        imgArr_ext = imgArr_o[y:y + h + int(hbuffer), x:x + w + int(wbuffer)]

        # Convert to grayscale for further processing
        imgArr_ext_gray = cv2.cvtColor(imgArr_ext, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to binarize the image
        imgArr_ext_pp = cv2.adaptiveThreshold(imgArr_ext_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

        # Apply median blur to reduce noise
        imgArr_ext_pp = cv2.medianBlur(imgArr_ext_pp, 13)

        # Predict the digit from the processed image
        predicted_digit = predict_digit(imgArr_ext_pp)
        print(f"Prediction Completed {predicted_digit}")

        # Draw bounding box around the detected meter region
        cv2.rectangle(imgArr_o, (x, y), (x + w + int(wbuffer), y + h + int(hbuffer)), (255, 0, 255), 10)
        
        # Break after the first contour for debugging (remove if you want to process all contours)
        break

    # Debugging: Save images for analysis
    if debug:
        os.makedirs('./traindata/meter_disp_ext/', exist_ok=True)
        os.makedirs('./traindata/mask/', exist_ok=True)
        os.makedirs('./traindata/meter_disp_bb/', exist_ok=True)
        os.makedirs('./traindata/meter_disp_ext_pp/', exist_ok=True)

        # Save the processed images
        cv2.imwrite(f'./traindata/meter_disp_ext/{img_path.split(".")[0]}_ext.png', imgArr_ext)
        cv2.imwrite(f'./traindata/mask/{img_path.split(".")[0]}_mask.png', mask)
        cv2.imwrite(f'./traindata/meter_disp_bb/{img_path.split(".")[0]}_bb.png', imgArr_o)
        cv2.imwrite(f'./traindata/meter_disp_ext_pp/{img_path.split(".")[0]}_pp.png', imgArr_ext_pp)

    print(f"{img_path} --> DONE")

# Main function to process all images
def main():
    path = "./meter_reading_images/"
    files = [file for file in os.listdir(path) if file.endswith('.png')]
    print(sorted(files))

    # Train and save the model (run this once)
    train_and_save_model()

    # Process all the meter images
    for meter in files:
        preprocess_image(meter, path)

if __name__ == "__main__":
    main()
