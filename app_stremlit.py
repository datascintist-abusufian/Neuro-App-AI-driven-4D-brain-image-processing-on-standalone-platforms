import os
import requests
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score

st.image("TAC_Brain_tumor_glioblastoma-Transverse_plane.gif",use_column_width=True)
st.sidebar.title("Brain Tumor Detection")
st.sidebar.write("User friendly, Public can test the MRI image segmentation accuracy")
uploaded_image = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], key="file_uploader")
uploaded_mask = st.sidebar.file_uploader("Choose the corresponding ground truth mask...", type=["jpg", "jpeg", "png"], key="mask_uploader")
st.title("4D AI Driven Neuro App")
st.write("This app demonstrates Brain Image segmentation and tumor detection and prevention using a web application.")
st.write("In this version of model design, the MRI image is displayed and the prediction is made as soon as the MRI image is uploaded. The accuracy is calculated and displayed only when the ground truth mask is uploaded.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.write("......Visualisation of Design and Coding Under Construction.........")
MODEL_PATH = 'BrainTumor10Epochs.h5'

def download_model(model_url, model_path):
    if not os.path.isfile(model_path):
        with st.spinner('Downloading model...'):
            response = requests.get(model_url)
            if response.status_code == 200:
                with open(model_path, 'wb') as file:
                    file.write(response.content)
            else:
                raise Exception(f"Error downloading the model: HTTP {response.status_code}")

@st.cache(allow_output_mutation=True)
def load_my_model():
    model_url = 'https://raw.githubusercontent.com/datascintist-abusufian/Neuro-App-AI-driven-4D-brain-image-processing-on-standalone-platforms/main/' + MODEL_PATH
    download_model(model_url, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert('RGB').resize((64, 64))
    st.image(img, caption="Uploaded MRI Image", width=400)
    try:
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred_mask = model.predict(img_array)
        if len(pred_mask.shape) == 4:
            pred_mask = np.squeeze(pred_mask, axis=[0, -1])
        elif len(pred_mask.shape) == 3:
            pred_mask = np.squeeze(pred_mask, axis=0)
        print(pred_mask.shape)
        if pred_mask is None:
            st.error("Model prediction returned None")
        else:
            threshold = 0.5
            binary_mask = (pred_mask > threshold).astype(np.uint8)
            binary_mask = np.squeeze(binary_mask)  # Remove batch and channel dimensions
            st.image(binary_mask, caption="Predicted Mask", width=400)
            # Tumor detection logic
            tumor_detected = binary_mask.max() > threshold
            if tumor_detected:
                st.write("Tumor detected. Please consult with a doctor.")
            else:
                st.write("No tumor detected. However, consult with a doctor for an accurate diagnosis.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if uploaded_mask is not None:
    mask = Image.open(uploaded_mask).convert('L').resize((64, 64))
    mask_array = np.array(mask)
    # Calculate accuracy on the flattened masks
    accuracy = accuracy_score(mask_array.flatten(), binary_mask.flatten())
    st.write(f"Prediction accuracy: {accuracy:.2f}")
