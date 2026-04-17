import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("PlantDNet.h5", compile=False)
print("Model loaded")

# ✅ Class labels (IMPORTANT)
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

# Friendly names
friendly_names = {
    'Pepper__bell___Bacterial_spot': 'Pepper Bacterial Spot',
    'Pepper__bell___healthy': 'Healthy Pepper Leaf',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Healthy Potato Leaf',
    'Tomato_Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato_Early_blight': 'Tomato Early Blight',
    'Tomato_Late_blight': 'Tomato Late Blight',
    'Tomato_Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato__Target_Spot': 'Tomato Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato_healthy': 'Healthy Tomato Leaf'
}

# Disease info
disease_info = {
    'Pepper Bacterial Spot': "Caused by bacteria. Avoid overhead watering.",
    'Healthy Pepper Leaf': "Your plant is healthy 🌿",
    'Potato Early Blight': "Fungal disease. Use fungicide.",
    'Potato Late Blight': "Serious disease. Act quickly.",
    'Healthy Potato Leaf': "Your plant is healthy 🌿",
    'Tomato Bacterial Spot': "Use copper sprays.",
    'Tomato Early Blight': "Remove infected leaves.",
    'Tomato Late Blight': "Highly destructive disease.",
    'Tomato Leaf Mold': "Reduce humidity.",
    'Tomato Septoria Leaf Spot': "Avoid splashing water.",
    'Tomato Spider Mites': "Use neem oil.",
    'Tomato Target Spot': "Apply fungicide.",
    'Tomato Yellow Leaf Curl Virus': "Remove infected plants.",
    'Tomato Mosaic Virus': "Avoid contamination.",
    'Healthy Tomato Leaf': "Your plant is healthy 🌿",
    'Unknown': "❌ Please upload a clear plant leaf image 🌿"
}

# Prediction function
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    return preds


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


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
        file.save(file_path)

        preds = model_predict(file_path)

        pred_index = np.argmax(preds[0])
        predicted_class = classes[pred_index]

        # 🔥 SMART CHECK (FIX)
        unknown, confidence = is_unknown(preds)

        if unknown:
            final_prediction = "Unknown"
            info = "❌ This does not look like a plant leaf. Please upload a clear leaf image."
        else:
            final_prediction = friendly_names.get(predicted_class, predicted_class)
            info = disease_info.get(final_prediction, "No info available")

        return render_template(
            "result.html",
            prediction=final_prediction,
            confidence=round(confidence, 2),
            filename=filename,
            info=info
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)