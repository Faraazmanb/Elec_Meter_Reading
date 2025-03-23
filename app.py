from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import io
import os
from dotenv import load_dotenv
# Import your prediction function
from prediction_algo import predict
load_dotenv()
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend for Matplotlib

# Initialize Flask app
app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")  # Get URI from .env

mongo = PyMongo(app)
db = mongo.db  # Get the database object
users_collection = db.users  # Example collection
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
app.secret_key = os.getenv("SECRET_KEY")

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": user_id})
    if user:
        return User(user_id)
    return None

# Home Page
@app.route('/')
def index():
    session.clear() 
    return render_template('loginreg.html')

@app.route('/index')
def indexx():
    return render_template('index.html')

# Model Page
@app.route('/model')
def model():
    product_description = "This is the product description from the backend."
    return render_template('upload.html', product_description=product_description)

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if mongo.db.users.find_one({"_id": username}):
            flash("Username already exists!", "error")
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({"_id": username, "password": hashed_password})
        
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    
    return render_template('loginreg.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = mongo.db.users.find_one({"_id": username})
        if user and check_password_hash(user["password"], password):
            login_user(User(username))
            flash("Login successful!", "success")
            return redirect(url_for('indexx'))
        else:
            flash("Invalid username or password.", "error")
            return redirect(url_for('login'))
    
    return render_template('loginreg.html')

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Dashboard (Protected Page)
# @app.route('/index')
# @login_required
# def dashboard():
#     return f"Welcome, {current_user.id}! <a href='/logout'>Logout</a>"

# File upload route
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
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


# Predict route
@app.route('/predict', methods=['POST'])
@login_required
def predict_image():
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
