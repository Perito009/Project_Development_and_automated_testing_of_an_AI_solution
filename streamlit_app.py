import streamlit as st
import pickle

MODEL_PATH = 'spam_model.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        return data['model'], data['vectorizer']

model, vectorizer = load_model()

st.title("Spam Classifier")
st.write("Enter a message to check if it's spam or ham.")

user_input = st.text_area("Message", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        st.success(f"Prediction: **{prediction}**")
