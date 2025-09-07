import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load trained CNN model
model = load_model('model.h5')
print(' Model loaded. Check http://127.0.0.1:5000/')

# Get class labels (from training generator)
# Make sure you saved this mapping during training step 7
# Example: np.save("class_indices.npy", train_generator.class_indices)
if os.path.exists("class_indices.npy"):
    labels = np.load("class_indices.npy", allow_pickle=True).item()
    labels = {v: k for k, v in labels.items()}  # reverse mapping
else:
    raise FileNotFoundError(" class_indices.npy not found! Save class indices during training.")

# Preprocess image for prediction
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))  # resize same as training
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Ensure uploads folder exists
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Prediction
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

        return f"Predicted: {predicted_label} ({confidence}%)"
    return None

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
