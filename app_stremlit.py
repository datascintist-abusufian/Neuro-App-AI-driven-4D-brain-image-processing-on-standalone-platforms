import os
import requests
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

st.image("TAC_Brain_tumor_glioblastoma-Transverse_plane.gif", use_column_width=True)

# Sidebar for file upload
st.sidebar.title("Brain Tumor Detection")
st.sidebar.write("User friendly, Public can test the MRI image segmentation accuracy")
uploaded_image = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], key="file_uploader")

# Title and description
st.title("4D AI Driven Neuro App")
st.write("This app demonstrates Brain Image segmentation and tumor detection and prevention using a web application.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.write("......Visualisation of Design and Coding Under Construction.........")

MODEL_PATH = 'BrainTumor10Epochs.h5'  # The name of the model file

# Function to download the model if it's not already on the filesystem
def download_model(model_url, model_path):
    if not os.path.isfile(model_path):
        with st.spinner('Downloading model...'):
            response = requests.get(model_url)
            if response.status_code == 200:
                with open(model_path, 'wb') as file:
                    file.write(response.content)
            else:
                raise Exception(f"Error downloading the model: HTTP {response.status_code}")

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_my_model():
    model_url = 'https://raw.githubusercontent.com/datascintist-abusufian/Neuro-App-AI-driven-4D-brain-image-processing-on-standalone-platforms/main/' + MODEL_PATH
    download_model(model_url, MODEL_PATH)
    model = load_model(MODEL_PATH)  # Load the model from the local path
    return model

model = load_my_model()

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert('RGB')
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    try:
        img_for_pred = img.resize((64, 64))  # Resize the image to match the model's expected input shape
        img_array = np.array(img_for_pred) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred_mask = model.predict(img_array)
        
        threshold = 0.5
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        
        # Create an RGBA image for the overlay with the mask as the alpha channel
        mask_colored = np.stack([binary_mask*0, binary_mask*255, binary_mask*0, binary_mask*255], axis=-1)
        overlay = Image.fromarray(mask_colored, mode='RGBA')
        img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        st.image(img_with_overlay, caption="Segmentation Result", use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")  # Display the actual error message
else:
    st.error("Please upload an image file.")
