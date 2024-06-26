# model_training.py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from src.feature_extraction import extract_features
from src.logger import logger

def prepare_data(data):
    try:
        X = np.array([np.array(extract_features(file_path)) for file_path in data['file_path']])
        y = data['emotion'].values

        label_encoder = LabelEncoder()
        y = to_categorical(label_encoder.fit_transform(y))

        return X, y, label_encoder

    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise e

def build_model(input_shape):
    try:
        model = Sequential([
            LSTM(256, input_shape=(input_shape, 1), return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(9, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise e

def train_model(model, X_train, X_test, y_train, y_test):
    try:
        history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                            validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
                            epochs=50, batch_size=32)
        return model, history

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise e


