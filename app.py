import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("leafedetection.h5")

# Define the labels corresponding to the classes
labels = [
    "Alstonia Scholaris diseased", "Alstonia Scholaris healthy",
    "Arjun diseased", "Arjun healthy",
    "Bael diseased", "Basil healthy",
    "Chinar diseased", "Chinar healthy",
    "Gauva diseased", "Gauva healthy",
    "Jamun diseased", "Jamun healthy",
    "Jatropha diseased", "Jatropha healthy",
    "Lemon diseased", "Lemon healthy",
    "Mango diseased", "Mango healthy",
    "Pomegranate diseased", "Pomegranate healthy",
    "Pongamia Pinnata diseased", "Pongamia Pinnata healthy"
]

def preprocess_image(img):
    """Load and preprocess the image."""
    img = img.resize((150, 150))  # Resize image to expected input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0, 1]
    return img_array

def classify_image(img):
    """Predict the class of the image."""
    img_array = preprocess_image(img)  # Preprocess the image

    try:
        predictions = model.predict(img_array)  # Get predictions from the model
        predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get class index
        return predicted_class_index
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit app
st.title("Leaf Disease Classification")
st.write("Upload a leaf image to classify its condition.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Classify the image
    if st.button("Classify"):
        predicted_class = classify_image(img)  # Classify the uploaded image
        if predicted_class is not None:
            st.success(f"The predicted class index is: {predicted_class}")
            st.success(f"The predicted class is: {labels[predicted_class]}")
