import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import streamlit as st
from PIL import Image

st.image("BrainTumor.gif", use_column_width=True)

# Function to load the model (cached)
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = load_model('BrainTumor10Epochs.h5')
    return model

# Function to get class name
def get_className(class_no):
    if class_no == 0:
        return 'Brain Tumor detected in this MRI Image. The patient needs to consult the Neuro Specialist.'
    elif class_no == 1:
        return 'Brain Tumor not detected in this MRI Image. The patient does not need to consult the Neuro specialist.'

# Loading the model
model = load_my_model()

# Streamlit UI
st.title('Brain Tumor Detection Web App')
uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    if uploaded_image.type in ["image/jpeg", "image/png", "image/jpg"]:
        try:
            # Reading the image
            img = Image.open(uploaded_image)
            img = img.resize((256, 256))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Making a prediction
            prediction = model.predict(img_array)
            classification = prediction.argmax()
            classification_text = get_className(classification)

            # Displaying results
            st.image(uploaded_image, caption="Uploaded MRI Image", use_column_width=True)
            st.write(classification_text)

        except Exception as e:
            st.error(f"Error occurred: {e}")
    else:
        st.error("Please upload an image file.")
