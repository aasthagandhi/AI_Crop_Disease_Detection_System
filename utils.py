import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ✅ 1. Load trained model
model = load_model("model/aasu_crop_model.keras")

# ✅ 2. Class names (IMPORTANT)
# 👉 Replace these with your actual folder names
# class_names = [
#     "Tomato_Late_blight",
#     "Tomato_Early_blight",
#     "Potato__Early_blight"
# ]

import os

class_names = sorted(os.listdir("C:/Users/rishu/desktop/crop-disease-app/dataset/crop_disease/PlantVillage"))

# ✅ 3. Disease info (can expand later)
disease_info = {
    "Tomato_Late_blight": {
        "cause": "Caused by water mold in cool, wet weather",
        "solution": "Apply copper fungicide and remove infected leaves"
    },
    "Tomato_Early_blight": {
        "cause": "Fungal infection due to humidity",
        "solution": "Use chlorothalonil spray"
    },
    "Potato__Early_blight": {
        "cause": "Fungal disease",
        "solution": "Remove affected leaves and apply fungicide"
    }
}

# ✅ 4. Prediction function
def predict_disease(image):

    # Resize image (same as training)
    img = cv2.resize(image, (224, 224))

    # Normalize (0–1)
    img = img / 255.0

    # Reshape for model
    img = np.reshape(img, (1, 224, 224, 3))

    # Predict
    prediction = model.predict(img)

    # Get index of highest probability
    class_index = np.argmax(prediction)

    # Get confidence
    confidence = float(np.max(prediction))

    # Get disease name
    disease = class_names[class_index]

    # Get extra info
    info = disease_info.get(disease, {})

    cause = info.get("cause", "No info available")
    solution = info.get("solution", "No solution available")

    return disease, confidence, cause, solution