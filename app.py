import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
import os

# Use __file__ if available, otherwise use the current working directory
current_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
model_file_path = os.path.join(current_directory, 'pneumonia_model.h5')

# Debug: Print the model file path
st.write("Model file path:", model_file_path)

# Check if the model file exists
if not os.path.exists(model_file_path):
    st.error(f"Model file not found at {model_file_path}")
else:
    st.write("Model file found. Loading model...")
    model = tf.keras.models.load_model(model_file_path)
    st.write("Model loaded successfully.")

# Streamlit App
st.title("Pneumonia Detection from CT Scans")

uploaded_file = st.file_uploader("Upload a CT scan image...", type=["jpeg", "jpg", "png", "pdf"])

def preprocess_image(image):
    if image.type == "application/pdf":
        images = convert_from_path(image)
        image = np.array(images[0])
    else:
        image = Image.open(image)
        image = np.array(image)

    # Resize the image to (150, 150)
    image = cv2.resize(image, (150, 150))

    # Convert grayscale to RGB if the image has only 1 channel
    if len(image.shape) == 2:  # Grayscale image (height, width)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
    elif image.shape[2] == 4:  # RGBA image (height, width, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Convert to 3-channel RGB

    # Add batch dimension and normalize pixel values
    image = np.expand_dims(image, axis=0)  # Shape: (1, 150, 150, 3)
    return image / 255.0

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    try:
        image = preprocess_image(uploaded_file)
        st.write("Image preprocessed. Making prediction...")
        prediction = model.predict(image)
        if prediction < 0.5:
            st.write("The scan is Normal")
        else:
            st.write("The scan shows Pneumonia")
        st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)
    except Exception as e:
        st.error(f"Error processing file: {e}")