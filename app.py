import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('PlantDNet.h5', compile=False)
print("Model loaded")

# Prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    return preds

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    f = request.files['file']
    file_path = os.path.join("uploads", secure_filename(f.filename))
    f.save(file_path)

    preds = model_predict(file_path, model)

    disease_class = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
    ]

    result = disease_class[np.argmax(preds[0])]
    return f"Prediction: {result}"

# Run app (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("PlantDNet.h5", compile=False)
print("Model loaded")

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

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        filename = secure_filename(file.filename)

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        preds = model_predict(file_path)

        disease_class = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite',
            'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
        ]

        result = disease_class[np.argmax(preds[0])]

        return result

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)