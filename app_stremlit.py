import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import streamlit as st
from werkzeug.utils import secure_filename

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')
st.title('Brain Tumor Detection Web App')

def get_className(class_no):
    if class_no == 0:
        return 'Brain Tumor detected in this MRI Image. The patient needs to consult the Neuro Specialist.'
    elif class_no == 1:
        return 'Brain Tumor not detected in this MRI Image. The patient does not need to consult the Neuro specialist.'

uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = image.load_img(uploaded_image, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    classification = prediction.argmax()
    classification_text = get_className(classification)
    st.image(uploaded_image, caption="Uploaded MRI Image", use_column_width=True)
    st.write(classification_text)
