"""
Model Training Script for Emotion Detection
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.config import *
from src.model import create_emotion_model, compile_model, get_model_summary
from src.data_loader import create_data_generators, print_dataset_info


def get_callbacks(model_save_path=MODEL_SAVE_PATH):
    """Create training callbacks"""
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    )
    
    return [checkpoint, early_stopping, reduce_lr]


def train_model(epochs=50, learning_rate=0.001):
    """Main training function"""
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    print("\n=== Loading Data ===")
    train_gen, val_gen, test_gen = create_data_generators()
    print_dataset_info(train_gen, val_gen, test_gen)
    
    print("\n=== Creating Model ===")
    model = create_emotion_model()
    model = compile_model(model, learning_rate=learning_rate)
    get_model_summary(model)
    
    print("\n=== Starting Training ===")
    callbacks = get_callbacks()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    return model, history


if __name__ == "__main__":
    model, history = train_model(epochs=50, learning_rate=0.001)
