# feature_extraction.py
import librosa
import numpy as np
from src.logger import logger

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
            features.append(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            features.append(mel)
        features = np.hstack(features)
        return features

    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        raise e


