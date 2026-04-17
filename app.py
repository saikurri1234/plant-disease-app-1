import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# 🔥 MobileNetV2 for filtering non-leaf images
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
plant_model = tf.keras.models.load_model("PlantDNet.h5", compile=False)
print("Plant disease model loaded")

leaf_filter_model = MobileNetV2(weights="imagenet")
print("Leaf filter model loaded")

# -----------------------------
# PLANT CLASSES
# -----------------------------
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
    'Tomato_healthy': "Your plant is healthy 🌿"
}

# -----------------------------
# 🔥 LEAF FILTER (IMPORTANT FIX)
# -----------------------------
def is_leaf_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = leaf_filter_model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    plant_words = ["leaf", "plant", "tree", "flower", "vine", "crop"]

    for _, label, _ in decoded:
        if any(word in label.lower() for word in plant_words):
            return True

    return False


# -----------------------------
# PLANT MODEL PREDICTION
# -----------------------------
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return plant_model.predict(x)


# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 🔥 STEP 1: FILTER NON-LEAF IMAGES
        if not is_leaf_image(file_path):
            return render_template(
                "result.html",
                prediction="Unknown (Not a Plant Leaf)",
                confidence=0,
                filename=filename,
                info="❌ This is not a plant leaf image. Please upload a leaf."
            )

        # 🔥 STEP 2: PLANT DISEASE PREDICTION
        preds = model_predict(file_path)

        pred_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]) * 100)

        predicted_class = classes[pred_index]
        final_prediction = friendly_names.get(predicted_class, predicted_class)

        return render_template(
            "result.html",
            prediction=final_prediction,
            confidence=round(confidence, 2),
            filename=filename,
            info=disease_info.get(final_prediction, "")
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)