# app.py - Spam Mail Detector Web App
from flask import Flask, render_template, request
import joblib
import os

# ✅ Define the Flask app here
app = Flask(__name__)

# Load trained model + vectorizer
MODEL_DIR = "models"
vect = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "spam_model.joblib"))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("message", "")
    if not text.strip():
        return render_template("result.html", prediction="⚠ Please enter a message.", prob=None)

    # Vectorize + predict
    X = vect.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    label = "🚨 Spam" if pred == 1 else "✅ Ham (Not Spam)"
    return render_template("result.html",
                           prediction=label,
                           prob=f"{prob*100:.2f}%",
                           message=text)

# 🚀 Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
