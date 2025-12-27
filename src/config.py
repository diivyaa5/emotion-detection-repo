"""
Configuration file for Emotion Detection model
"""
import os

DATASET_PATH = '/content/images'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
TEST_DIR = os.path.join(DATASET_PATH, 'validation')

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NUM_CLASSES = len(EMOTIONS)

RANDOM_SEED = 42

MODEL_SAVE_PATH = 'models/emotion_detection_model.h5'
