# app.py - Spam Mail Detector (Streamlit with UI Enhancements)
import streamlit as st
import joblib
import os

# ========== Page Config ==========
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== Load Model ==========
MODEL_DIR = "models"
vect = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "spam_model.joblib"))

# ========== App UI ==========
st.markdown(
    """
    <style>
        body {
            background-color: #f4f9ff;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📧 Spam Mail Detector")
st.write("Check whether a message is **Spam** or **Ham (Not Spam)**")

# User input
text = st.text_area("✍️ Enter your message here:", "")

if st.button("🔍 Predict"):
    if not text.strip():
        st.warning("⚠ Please enter a message to analyze.")
    else:
        # Vectorize + predict
        X = vect.transform([text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()

        # Results
        if pred == 1:
            st.error("🚨 **Spam Detected!**")
        else:
            st.success("✅ **This looks like Ham (Not Spam)**")

        # Confidence bar
        st.write("### Confidence Level")
        st.progress(int(prob * 100))
        st.write(f"**{prob*100:.2f}%**")

        # Show original message
        st.write("### Your Message:")
        st.info(text)




