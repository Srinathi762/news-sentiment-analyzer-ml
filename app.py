# app.py

import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

# Predict sentiment
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return pred

# Map label
def label_with_emoji(label_id):
    label_map = {
        0: "Negative 😡",
        1: "Neutral 😐",
        2: "Positive 😊"
    }
    return label_map.get(label_id, "Unknown ❓")

# ----------------- Streamlit App ------------------

# Page Config
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

# Header
st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a sentence below and detect the sentiment in real-time.</p>", unsafe_allow_html=True)

# Input
user_input = st.text_area("✍️ Enter your text:", placeholder="Type something like 'I am very happy with the service.'")

# Button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = predict_sentiment(user_input)
        result = label_with_emoji(prediction)

        # Show result
        st.success(f"**Predicted Sentiment:** {result}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
