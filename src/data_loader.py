"""
Data loading and preprocessing for Emotion Detection
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import *


def create_data_generators():
    """Create training, validation, and test data generators"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def count_images(directory):
    """Count images in each emotion category"""
    counts = {}
    for emotion in EMOTIONS:
        emotion_path = os.path.join(directory, emotion)
        if os.path.exists(emotion_path):
            counts[emotion] = len(os.listdir(emotion_path))
        else:
            counts[emotion] = 0
    return counts


def print_dataset_info(train_generator, validation_generator, test_generator):
    """Print dataset statistics"""
    train_counts = count_images(TRAIN_DIR)
    test_counts = count_images(TEST_DIR)
    
    print("\n=== Training Set Distribution ===")
    for emotion, count in train_counts.items():
        print(f"{emotion.capitalize()}: {count} images")
    print(f"\nTotal Training Images: {sum(train_counts.values())}")
    
    print("\n=== Test Set Distribution ===")
    for emotion, count in test_counts.items():
        print(f"{emotion.capitalize()}: {count} images")
    print(f"\nTotal Test Images: {sum(test_counts.values())}")
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"\nClass indices: {train_generator.class_indices}")


def visualize_sample_images(train_generator, num_images=8):
    """Visualize sample augmented training images"""
    sample_batch, sample_labels = next(train_generator)
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(sample_batch))):
        axes[i].imshow(sample_batch[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        emotion_idx = np.argmax(sample_labels[i])
        emotion_name = list(train_generator.class_indices.keys())[emotion_idx]
        axes[i].set_title(f'{emotion_name.capitalize()}', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle('Augmented Training Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
