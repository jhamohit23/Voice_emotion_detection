import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils import get_logger
import pandas as pd

logger = get_logger(__name__)

def extract_features(file_path):
    """
    Extracts features from an audio file.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def load_data(ravdess_path, tess_path):
    try:
        X, y = [], []

        # Process RAVDESS dataset
        for actor in os.listdir(ravdess_path):
            if actor.startswith('Actor_') and int(actor[-2:]) % 2 == 0:  # Even numbers are female
                for file in os.listdir(os.path.join(ravdess_path, actor)):
                    emotion = file.split('-')[2]
                    if emotion == '01':  # Neutral
                        continue
                    emotion_map = {'02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
                    emotion_label = emotion_map[emotion]
                    file_path = os.path.join(ravdess_path, actor, file)
                    X.append(file_path)
                    y.append(emotion_label)

        # Process TESS dataset
        for file in os.listdir(tess_path):
            if file.endswith('.wav'):
                emotion = file.split('_')[-1].split('.')[0]
                if emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                    file_path = os.path.join(tess_path, file)
                    X.append(file_path)
                    y.append(emotion)

        data = pd.DataFrame({'file_path': X, 'emotion': y})
        return data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
