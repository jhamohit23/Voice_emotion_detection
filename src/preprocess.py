# data_processing.py
import os
import pandas as pd
from src.logger import logger

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

