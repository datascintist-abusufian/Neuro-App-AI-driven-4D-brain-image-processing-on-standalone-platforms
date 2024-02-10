import os
import requests
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score

st.image("TAC_Brain_tumor_glioblastoma-Transverse_plane.gif", width=400)
st.sidebar.title("Brain Tumor Detection")
st.sidebar.write("User friendly, Public can test the MRI image segmentation accuracy")
uploaded_image = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], key="file_uploader")
uploaded_mask = st.sidebar.file_uploader("Choose the corresponding ground truth mask...", type=["jpg", "jpeg", "png"], key="mask_uploader")
st.title("4D AI Driven Neuro App")
st.write("This app demonstrates Brain Image segmentation and tumor detection and prevention using a web application.")
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

if uploaded_image is not None and uploaded_mask is not None:
    img = Image.open(uploaded_image).convert('RGB').resize((64, 64))
    mask = Image.open(uploaded_mask).convert('L').resize((64, 64))
    st.image(img, caption="Uploaded MRI Image", width=400)
    
    try:
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_mask = model.predict(img_array)
    if pred_mask is None:
        st.error("Model prediction returned None")
        return
    threshold = 0.5
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    mask_colored = np.stack([binary_mask*255, binary_mask*0, binary_mask*0, binary_mask*255], axis=-1)  # Red color for tumor area
    overlay = Image.fromarray(mask_colored.squeeze(), mode='RGBA')  # Remove single-dimensional entries from the shape of an array
    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay)
    st.image(img_with_overlay, caption="Segmentation Result", use_column_width=True)
    mask_array = np.array(mask)
    if mask_array.shape != binary_mask.shape:
        st.error("The shapes of the ground truth mask and the predicted mask do not match")
        return
    accuracy = accuracy_score(mask_array.flatten(), binary_mask.flatten())
    st.write(f"Prediction accuracy: {accuracy:.2f}")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
