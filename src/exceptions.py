class DataLoadingError(Exception):
    """
    Exception raised for errors during data loading.
    """
    pass

class FeatureExtractionError(Exception):
    """
    Exception raised for errors during feature extraction.
    """
    pass

class ModelTrainingError(Exception):
    """
    Exception raised for errors during model training.
    """
    pass

class ModelLoadingError(Exception):
    """
    Exception raised for errors during model loading.
    """
    pass

class PredictionError(Exception):
    """
    Exception raised for errors during prediction.
    """
    pass