import os
import numpy as np
import cv2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with tf.keras model.save()
MODEL_PATH = './models/cnn_model.keras'
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = load_model(MODEL_PATH, compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    z_img = cv2.imread(img_path)
    z_img = cv2.resize(z_img, (70, 70)) / 255.0
    z_img = z_img.reshape(1, z_img.shape[0], z_img.shape[1], z_img.shape[2])

    
    preds = model.predict(z_img)
    preds = np.argmax(preds, axis = 1)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path);
        # Process your result for human
        print(preds);
        if preds == 1:
            prediction = "1"
        else:
            prediction = "0"              
        return prediction
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')