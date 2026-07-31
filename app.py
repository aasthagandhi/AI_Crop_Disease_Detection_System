from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from utils import predict_disease

app = Flask(__name__)


import os

print("Current Directory:", os.getcwd())
print("Model Exists:", os.path.exists("model/aasu_crop_model.keras"))

# Home route
@app.route('/')
def home():
    return render_template("index.html")


# About route
@app.route('/about')
def about():
    return render_template("about.html")


# Contact route
@app.route('/contact')
def contact():
    return render_template("contact.html")


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    print("=====PREDICT ROUTE HIT =====",flush=True)
    try:
        if 'image' not in request.files:
            return "No image uploaded."

        file = request.files['image']

        if file.filename == '':
            return "Please select an image."

        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        disease, confidence, cause, solution, prevention = predict_disease(img)

        return render_template(
            "index.html",
            disease=disease,
            confidence=round(confidence * 100, 2),
            cause=cause,
            solution=solution,
            prevention=prevention
        )

    except Exception:
        import traceback
        return f"<pre>{traceback.format_exc()}</pre>"


# Chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.form['message'].lower()

    response = "Sorry, I don't understand your question."

    if "late blight" in user_msg:
        response = "Late blight is caused by fungus. Use copper fungicide and remove infected leaves."

    elif "early blight" in user_msg:
        response = "Early blight can be treated using chlorothalonil spray."

    elif "solution" in user_msg:
        response = "Please mention the disease name for proper solution."

    elif "hello" in user_msg or "hi" in user_msg:
        response = "Hello! How can I help you with crop diseases?"

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)