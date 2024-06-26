import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import load_data
from src.feature_extraction import extract_features
from src.model_training import prepare_data, build_model, train_model
from src.logger import logger

# Load data
ravdess_path = os.path.join("data", "ravdess")
tess_path = os.path.join("data", "tess")
data = load_data(ravdess_path, tess_path)

# Prepare data
X, y, label_encoder = prepare_data(data)

# Create the 'temp' directory if it doesn't exist
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(temp_dir, exist_ok=True)

# Build model
input_shape = X.shape[1]
model = build_model(input_shape)

# Load model weights
model_path = os.path.join("model", "best_model.keras")
if os.path.exists(model_path):
    model.load_weights(model_path)
else:
    logger.info("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, _ = train_model(model, X_train, X_test, y_train, y_test)
    model.save_weights(model_path)
    logger.info("Model training completed.")

# Streamlit app
st.title("Emotion Detection from Voice")

# Upload voice file
uploaded_file = st.file_uploader("Upload a voice file", type=["wav"])

if uploaded_file is not None:
    # Extract features
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    features = extract_features(file_path)
    features = (features - X.mean(axis=0)) / X.std(axis=0)
    features = features.reshape(1, -1, 1)

    # Predict emotion
    prediction = model.predict(features)
    emotion_label = label_encoder.inverse_transform(prediction.argmax(axis=1))[0]

    # Display prediction
    st.success(f"Predicted emotion: {emotion_label}")

    # Remove temporary file
    os.remove(file_path)