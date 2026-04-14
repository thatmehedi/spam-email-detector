import streamlit as st
import joblib
import re
import string

# Load models from models folder
nb_model = joblib.load('models/nb_spam_model.pkl')
knn_model = joblib.load('models/knn_spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title("📧 Spam Email Detector")

model_choice = st.radio("Select Model:", ["Naive Bayes", "KNN"])

email_text = st.text_area("Paste email here:", height=150)

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if st.button("Check Email"):
    if email_text:
        cleaned = clean_text(email_text)
        vectorized = vectorizer.transform([cleaned])
        
        if model_choice == "Naive Bayes":
            result = nb_model.predict(vectorized)[0]
        else:
            result = knn_model.predict(vectorized)[0]
        
        if result == 0:
            st.success("✅ NOT SPAM (Ham)")
        else:
            st.error("⚠️ SPAM DETECTED!")
    else:
        st.warning("Please paste an email")