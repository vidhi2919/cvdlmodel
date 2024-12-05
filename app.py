from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = tf.keras.models.load_model('anti_spoofing_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save the file
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        prediction_value = float(prediction[0][0])
        
        is_real = prediction_value > 0.5
        confidence = prediction_value if is_real else 1 - prediction_value
        
        result = {
            'is_real': bool(is_real),
            'confidence': float(confidence),
            'raw_score': float(prediction_value),
            'image_path': f'/uploads/{filename}'
        }
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify(result)
        else:
            return render_template('result.html', **result)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        prediction_value = float(prediction[0][0])
        
        is_real = prediction_value > 0.5
        confidence = prediction_value if is_real else 1 - prediction_value
        
        return jsonify({
            'is_real': bool(is_real),
            'confidence': float(confidence),
            'raw_score': float(prediction_value)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)