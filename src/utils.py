import logging
import tensorflow as tf
import os

# Create the logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

from tensorflow.keras.models import load_model as keras_load_model

model_path = "C:/Users/Dell/OneDrive/Desktop/Voice_emotion_detection/model/best_model.keras"
def load_model(model, model_path):
    model = keras_load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def get_logger(name):
    """
    Creates and returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(logs_dir, 'app.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def load_model(model, model_path):
    """
    Loads the pre-trained model weights.
    """
    logger = get_logger(__name__)
    logger.info(f"Loading model from {model_path}")
    model.load_weights(model_path)
    return model