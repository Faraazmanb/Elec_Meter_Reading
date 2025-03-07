from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import os
from prediction_algo import predict  # Import your prediction function
import matplotlib

matplotlib.use('agg')  # Use a non-interactive backend for Matplotlib

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')  # Static file handling
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Model upload page route
@app.route('/model')
def model():
    product_description = "This is the product description from the backend."
    return render_template('upload.html', product_description=product_description)

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file upload, processes the image, and returns a prediction result.
    """
    try:
        if 'upload_file' not in request.files:
            app.logger.error("No file uploaded.")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['upload_file']
        if file.filename == '':
            app.logger.error("Empty file uploaded.")
            return jsonify({"error": "File is empty"}), 400

        app.logger.info(f"File received: {file.filename}")

        # Load image into PIL
        file_data = file.read()
        pil_image = Image.open(io.BytesIO(file_data))

        # Convert image to RGB if it's RGBA
        if pil_image.mode == 'RGBA':
            app.logger.info("Converting RGBA image to RGB mode.")
            pil_image = pil_image.convert('RGB')

        # Save to uploads folder
        temp_image_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
        pil_image.save(temp_image_path, format='JPEG')

        # Run the prediction function
        app.logger.info("Running prediction...")
        result = predict(temp_image_path, model_version="best")

        # Remove the temporary file after prediction
        os.remove(temp_image_path)
        app.logger.info("Temporary file removed.")

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error processing file: {e}")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

# Predict route (alternative)
@app.route('/predict', methods=['POST'])
def predict_image():
    """
    Endpoint to handle image uploads and perform prediction.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save to uploads folder
    temp_image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(temp_image_path)

    # Run the prediction
    result = predict(temp_image_path)

    # Delete the temporary file
    os.remove(temp_image_path)

    return jsonify(result)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
