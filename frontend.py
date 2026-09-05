import streamlit as st
import requests

RF_URL = "http://127.0.0.1:8001/predict"

st.title("AI vs Real Video Classifier")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(RF_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]

            st.success(f"Prediction: {prediction}")

        else:
            st.error("Prediction failed.")
            st.write(response.text)
