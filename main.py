import json
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained Keras model
model = load_model("plant_disease_prediction_model.h5")

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {int(v): k for v, k in class_indices.items()}  # keys must be int

# Preprocess image for prediction
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    return img_array

# Predict class
def predict_image_class(model, image_file, idx_to_class):
    preprocessed_img = preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = idx_to_class.get(predicted_index, "Unknown")
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence

# Streamlit app
st.title("🌿 Plant Disease Classifier")
st.markdown("#### Powered by a Convolutional Neural Network (CNN)")


uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
       st.image(image.resize((150, 150)), caption="Uploaded Image", use_container_width=True)


    with col2:
        if st.button("Classify"):
            predicted_class, confidence = predict_image_class(model, uploaded_image, idx_to_class)
            st.success(f"✅ Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
