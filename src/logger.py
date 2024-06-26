# logger.py
import logging
import os

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=os.path.join("logs", "emotion_detection.log"), level=logging.DEBUG, format=LOG_FORMAT)

logger = logging.getLogger(__name__)