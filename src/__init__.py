"""
Emotion Detection Package
"""
from .config import *
from .model import create_emotion_model, compile_model
from .data_loader import create_data_generators
from .train import train_model
from .evaluate import main_evaluate
from .real_time_detection import EmotionDetector

__version__ = "1.0.0"
