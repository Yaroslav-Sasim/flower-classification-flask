from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from model import build_model
from settings import IMG_SIZE, DATA_DIR

app = Flask(__name__)

weights_path = os.path.join(DATA_DIR, "FLOWERS.weights.h5")
model = build_model((IMG_SIZE, IMG_SIZE, 3), num_classes=5)
model.load_weights(weights_path)

# Список классов
class_names = ['Маргарита', 'Одуванчик', 'Роза', 'Подсолнух', 'Тюльпан']

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]
        return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
