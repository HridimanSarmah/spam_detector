# app.py - Spam Mail Detector (Streamlit)
import streamlit as st
import joblib
import os

# Load trained model + vectorizer
MODEL_DIR = "models"
vect = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "spam_model.joblib"))

# App UI
st.title("📧 Spam Mail Detector")
st.write("Enter a message below to check if it's **Spam** or **Ham (Not Spam)**")

# Input text
text = st.text_area("✍️ Message", "")

if st.button("🔍 Predict"):
    if not text.strip():
        st.warning("⚠ Please enter a message.")
    else:
        # Vectorize + predict
        X = vect.transform([text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()

        label = "🚨 Spam" if pred == 1 else "✅ Ham (Not Spam)"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: **{prob*100:.2f}%**")


