import os
import io
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('modell.h5')

app.static_folder = "tampilan"

@app.route('/')
def index():
    return render_template('index.html')

import base64

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    image = request.files['image']
    img = Image.open(image)
    img = img.resize((150, 150))
    img = img.convert('RGB')  # Convert the image to RGB

    # Convert the image to a NumPy array
    img = np.array(img)
    img = img / 255.0
    img = img.reshape((150, 150, 3))  # Reshape the array
    img = (img * 255).astype(np.uint8)  # Convert the data type

    prediction = model.predict(np.expand_dims(img, axis=0))
    probability = prediction[0][0]
    label = 'Ripe Apple' if probability >= 0.5 else 'Raw Apple'
    percentage = f'{probability * 100:.2f}%'

    # Convert image to PIL Image
    img_pil = Image.fromarray(img)

    # Convert image to base64 string
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('predict.html', image=image_base64, label=label, percentage=percentage)


@app.route('/predict_capture', methods=['POST'])
def predict_capture():
    # Read the base64-encoded image from the hidden input field
    image_data = request.form['image']
    image_data = image_data.replace('data:image/jpeg;base64,', '')

    # Convert the base64 data to a PIL Image object
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Resize the image
    # Convert the image to RGB
    image = image.convert('RGB')

# Resize the image
    image = image.resize((150, 150))

# Convert the image to a numpy array
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    
    
    # Perform prediction using the model
    prediction = model.predict(img)
    probability = prediction[0][0]
    label = 'Ripe Apple' if probability >= 0.5 else 'Raw Apple'
    percentage = f'{probability * 100:.2f}%'

    # Convert image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('predict.html', image=image_base64, label=label, percentage=percentage)



if __name__ == '__main__':
    app.run()
