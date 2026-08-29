import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI-Based Fake News Detection System")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    if news.strip():

        vec_text = vectorizer.transform([news])
        prediction = model.predict(vec_text)[0]

        if prediction == 1:
            st.error("FAKE NEWS")
        else:
            st.success("REAL NEWS")

    else:
        st.warning("Please enter a news article.")