"""
Model Evaluation Script for Emotion Detection
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.config import *
from src.data_loader import create_data_generators


def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_generator):
    """Evaluate model on test data"""
    
    print("\n=== Evaluating Model ===")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    true_classes = test_generator.classes
    
    return predicted_classes, true_classes, test_accuracy


def plot_confusion_matrix(true_classes, predicted_classes):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(true_classes, predicted_classes):
    """Print detailed classification report"""
    
    print("\n=== Classification Report ===")
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=EMOTIONS,
        digits=4
    )
    print(report)


def main_evaluate(model_path=MODEL_SAVE_PATH):
    """Main evaluation function"""
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    print("Loading test data...")
    _, _, test_gen = create_data_generators()
    
    predicted_classes, true_classes, accuracy = evaluate_model(model, test_gen)
    
    plot_confusion_matrix(true_classes, predicted_classes)
    print_classification_report(true_classes, predicted_classes)
    
    return accuracy


if __name__ == "__main__":
    main_evaluate()
