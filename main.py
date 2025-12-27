"""
Main entry point for Emotion Detection project
"""
import argparse
import sys
from src.train import train_model
from src.evaluate import main_evaluate
from src.real_time_detection import main as run_detection


def main():
    """Main function with CLI arguments"""
    parser = argparse.ArgumentParser(description='Emotion Detection System')
    
    parser.add_argument(
        'mode',
        choices=['train', 'evaluate', 'detect'],
        help='Mode to run: train, evaluate, or detect'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for training (default: 0.001)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/emotion_detection_model.h5',
        help='Path to save/load model (default: models/emotion_detection_model.h5)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=== Training Mode ===")
        train_model(epochs=args.epochs, learning_rate=args.learning_rate)
        
    elif args.mode == 'evaluate':
        print("=== Evaluation Mode ===")
        main_evaluate(model_path=args.model_path)
        
    elif args.mode == 'detect':
        print("=== Real-time Detection Mode ===")
        run_detection()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
