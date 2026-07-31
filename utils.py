import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/aasu_crop_model.keras")


# Class Names


class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


# Disease Information


disease_info = {

    "Pepper__bell___Bacterial_spot":{
        "cause":"Bacterial infection",
        "solution":"Use copper based bactericide and remove infected leaves.",
        "prevention":"Use disease free seeds and avoid overhead irrigation."
    },

    "Pepper__bell___healthy":{
        "cause":"Healthy Plant",
        "solution":"No treatment required.",
        "prevention":"Continue proper irrigation and fertilization."
    },

    "Potato___Early_blight":{
        "cause":"Fungal disease",
        "solution":"Apply fungicide and remove infected leaves.",
        "prevention":"Crop rotation and proper field sanitation."
    },

    "Potato___Late_blight":{
        "cause":"Caused by Phytophthora infestans.",
        "solution":"Apply copper fungicide and destroy infected plants.",
        "prevention":"Avoid excessive moisture and use resistant varieties."
    },

    "Potato___healthy":{
        "cause":"Healthy Plant",
        "solution":"No treatment required.",
        "prevention":"Maintain healthy soil and regular care."
    },

    "Tomato_Bacterial_spot":{
        "cause":"Bacterial infection",
        "solution":"Use copper spray and remove infected leaves.",
        "prevention":"Use certified seeds and avoid leaf wetness."
    },

    "Tomato_Early_blight":{
        "cause":"Fungal infection",
        "solution":"Spray chlorothalonil fungicide.",
        "prevention":"Crop rotation and proper spacing."
    },

    "Tomato_Late_blight":{
        "cause":"Water mold infection",
        "solution":"Apply fungicide immediately.",
        "prevention":"Avoid excess watering and improve air circulation."
    },

    "Tomato_Leaf_Mold":{
        "cause":"Fungal disease in humid weather.",
        "solution":"Apply recommended fungicide.",
        "prevention":"Reduce humidity and improve ventilation."
    },

    "Tomato_Septoria_leaf_spot":{
        "cause":"Fungal infection",
        "solution":"Remove infected leaves and spray fungicide.",
        "prevention":"Avoid overhead watering."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite":{
        "cause":"Spider mite infestation",
        "solution":"Use miticide or neem oil spray.",
        "prevention":"Maintain proper humidity and monitor regularly."
    },

    "Tomato__Target_Spot":{
        "cause":"Fungal disease",
        "solution":"Apply fungicide.",
        "prevention":"Remove infected debris."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus":{
        "cause":"Virus spread by whiteflies.",
        "solution":"Control whiteflies and remove infected plants.",
        "prevention":"Use resistant varieties and insect nets."
    },

    "Tomato__Tomato_mosaic_virus":{
        "cause":"Viral infection",
        "solution":"Remove infected plants.",
        "prevention":"Disinfect tools and avoid tobacco contamination."
    },

    "Tomato_healthy":{
        "cause":"Healthy Plant",
        "solution":"No treatment required.",
        "prevention":"Continue proper plant care."
    }

}


# Prediction Function


def predict_disease(image):

    img = cv2.resize(image, (224,224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    disease = class_names[class_index]

    info = disease_info.get(
        disease,
        {
            "cause":"No information available.",
            "solution":"No information available.",
            "prevention":"No information available."
        }
    )

    return (
        disease,
        confidence,
        info["cause"],
        info["solution"],
        info["prevention"]
    )