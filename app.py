from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from utils import predict_disease

app = Flask(__name__)

# ✅ Home route (check server)
@app.route('/')
def home():
    return render_template("index.html")

# about route
@app.route('/about')
def about():
    return render_template("about.html")

#contact route
@app.route('/contact')
def contact():
    return render_template("contact.html")


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # 1. Get image file
    file = request.files['image']

    # 2. Convert file to image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    print("Image received")

    # 3. Predict
    disease, confidence, cause, solution, prevention = predict_disease(img)
    print("Prediction successful")

    # 4. Return result
    return render_template(
        "index.html",
        disease=disease,
        confidence=round(confidence * 100, 2),
        cause=cause,
        solution=solution,
        prevention=prevention
    )


# chatbot
@app.route('/chat', methods=['POST'])
def chat():

    user_msg = request.form['message'].lower()

    response = "Sorry, I don't understand your question."

    # Basic chatbot logic
    if "late blight" in user_msg:
        response = "Late blight is caused by fungus. Use copper fungicide and remove infected leaves."

    elif "early blight" in user_msg:
        response = "Early blight can be treated using chlorothalonil spray."

    elif "solution" in user_msg:
        response = "Please mention the disease name for proper solution."

    elif "hello" in user_msg or "hi" in user_msg:
        response = "Hello! How can I help you with crop diseases?"

    return jsonify({"response": response})


# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)