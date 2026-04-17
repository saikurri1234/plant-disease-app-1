import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("PlantDNet.h5", compile=False)
print("Model loaded")

# Classes (✅ FIX ADDED HERE)
classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Disease info (GLOBAL for clean code)
disease_info = {
    'Pepper__bell___Bacterial_spot': "Caused by bacteria. Remove infected leaves and avoid overhead watering.",
    'Pepper__bell___healthy': "Your plant is healthy 🌿",
    'Potato___Early_blight': "Fungal disease. Use fungicide and remove affected leaves.",
    'Potato___Late_blight': "Serious fungal disease. Remove plant and apply fungicide immediately.",
    'Potato___healthy': "Your plant is healthy 🌿",
    'Tomato_Bacterial_spot': "Avoid wet leaves. Use copper-based sprays.",
    'Tomato_Early_blight': "Remove infected leaves and use fungicide.",
    'Tomato_Late_blight': "Highly destructive. Remove plant and treat nearby plants.",
    'Tomato_Leaf_Mold': "Improve air circulation and reduce humidity.",
    'Tomato_Septoria_leaf_spot': "Remove infected leaves and avoid splashing water.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Use neem oil or insecticidal soap.",
    'Tomato__Target_Spot': "Apply fungicide and remove affected leaves.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Spread by whiteflies. Remove infected plants.",
    'Tomato__Tomato_mosaic_virus': "Avoid handling plants after tobacco use. Remove infected plants.",
    'Tomato_healthy': "Your plant is healthy 🌿"
}

# Prediction function
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    return preds


# Home page
@app.route('/')
def index():
    return render_template("index.html")


# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file
        file.save(file_path)

        # Predict
        preds = model_predict(file_path)

        predicted_class = classes[np.argmax(preds[0])]

        # Confidence score
        confidence = round(np.max(preds[0]) * 100, 2)

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=confidence,
            filename=filename,
            info=disease_info.get(predicted_class, "No info available")
        )

    except Exception as e:
        return f"Error: {str(e)}"


# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)