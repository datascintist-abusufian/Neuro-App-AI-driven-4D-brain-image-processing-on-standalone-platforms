import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import streamlit as st
from PIL import Image 
import requests  # Ensure requests is imported

st.image("TAC_Brain_tumor_glioblastoma-Transverse_plane.gif", use_column_width=True)
# Sidebar for file upload
st.sidebar.title("Brain Tumor Detection")
st.sidebar.write ("User friendly, Public can test the MRI image segmentation accuracy")
uploaded_image = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Title and description
st.title("4D AI Driven Neuro App")
st.write("This app demonstrates Brain Image segmentation and tumor detection and prevention using web application.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.write( " ......Visualisation of Design and Coding Under Construction.........")


# Correctly download and then load the model
def download_model(url, model_name):
    """
    Download the model from a given URL if it's not already in the cache.
    """
    if not os.path.isfile(model_name):
        with st.spinner(f'Downloading {model_name}...'):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(model_name, 'wb') as f:
                    f.write(r.content)
            else:
                raise Exception(f"Error downloading the model: HTTP {r.status_code}")
    return model_name
    
# Function to load the model (cached)
@st.cache(allow_output_mutation=True)
def load_my_model():
    model_url = 'https://raw.githubusercontent.com/datascintist-abusufian/Neuro-App-AI-driven-4D-brain-image-processing-on-standalone-platforms/main/BrainTumor10Epochs.h5'
    model_path = download_model(model_url, 'BrainTumor10Epochs.h5')
    model = load_model(model_path)  # Corrected to load from the local path
    return model

# Function to get class name
def get_className(class_no):
    if class_no == 0:
        return 'Brain Tumor detected in this MRI Image. The patient needs to consult the Neuro Specialist.'
    elif class_no == 1:
        return 'Brain Tumor not detected in this MRI Image. The patient does not need to consult the Neuro specialist.'

# Loading the model
model = load_my_model()
st.title("Brain Tumor Detection 4D Brain MRI Imaging")

# Streamlit UI for showing segmentation results
if uploaded_image is not None:
    if uploaded_image.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Display the uploaded image
        img = Image.open(uploaded_image).convert('RGB')
        st.image(img, caption="Uploaded MRI Image", use_column_width=True)
    
        try:
            # Reading the image
            img = Image.open(uploaded_image)
            img = img.resize((256, 256))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalizing
            img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
            
            # Predict the segmentation mask
            pred_mask = model.predict(img_array)[0]  # Assuming the model returns a batch
            
            # For demonstration, assuming the segmentation mask is the direct output
            # Convert segmentation mask back to image
            # Note: This step highly depends on your model's specific output format
            mask = np.argmax(segmentation_mask, axis=-1)
            mask = np.squeeze(mask)  # Remove batch dimension
            pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold the predictions to get binary mask
            mask_img = Image.fromarray(pred_mask.squeeze() * 255)  # Convert to an image
            
            # Displaying original and segmented images
            st.image(uploaded_image, caption="Uploaded MRI Image", use_column_width=True)
            st.image(mask_img, caption="Segmentation Result", use_column_width=True)
            
            # If you have specific accuracy or other metric, display it here
            # st.write("Detection Accuracy: XX%")

        except Exception as e:
            st.error(f"Error occurred: {e}")
    else:
        st.error("Please upload an image file.")
