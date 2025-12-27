# Real-Time Emotion Detection

A deep learning project for detecting emotions from facial expressions in real-time using Convolutional Neural Networks (CNN).

## Features

- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Deep CNN Architecture**: Custom CNN with batch normalization and dropout
- **Real-time Detection**: Webcam-based emotion detection
- **Data Augmentation**: Enhanced training with image augmentation
- **Model Evaluation**: Comprehensive metrics and visualizations

## Project Structure

```
emotion-detection-repo/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── model.py               # CNN model architecture
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation and metrics
│   └── real_time_detection.py # Real-time webcam detection
├── models/                    # Saved models directory
├── data/                      # Dataset directory
├── notebooks/                 # Jupyter notebooks
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the FER2013 emotion dataset or your preferred emotion dataset and place it in the `data/` directory with the following structure:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## Usage

### Training the Model

```bash
python -m src.train
```

This will:
- Load and preprocess the dataset
- Create data augmentation pipelines
- Build and compile the CNN model
- Train the model with callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
- Save the best model to `models/emotion_detection_model.h5`

### Evaluating the Model

```bash
python -m src.evaluate
```

This will:
- Load the trained model
- Evaluate on test data
- Generate confusion matrix
- Print classification report with precision, recall, and F1-scores

### Real-time Emotion Detection

```bash
python -m src.real_time_detection
```

This will:
- Load the trained model
- Open your webcam
- Detect faces and predict emotions in real-time
- Display bounding boxes with emotion labels and confidence scores
- Press 'q' to quit

## Model Architecture

The CNN model consists of:
- 4 Convolutional blocks with batch normalization and dropout
- Progressive channel increase (32 → 64 → 128 → 256)
- 2 Fully connected layers (512 → 256)
- Softmax output layer for 7 emotion classes

**Total Parameters**: ~2.5M

## Configuration

Edit `src/config.py` to customize:
- Dataset paths
- Image dimensions (default: 48x48)
- Batch size
- Emotion classes
- Model save path

## Dataset

This project is designed to work with facial emotion datasets such as:
- FER2013
- CK+
- JAFFE
- Custom datasets

Ensure your dataset follows the directory structure mentioned in the Installation section.

## Results

The model achieves competitive accuracy on the FER2013 dataset:
- Training Accuracy: ~70-75%
- Validation Accuracy: ~65-70%
- Test Accuracy: ~65-70%

Results may vary based on dataset quality and hyperparameters.

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.7+
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## License

This project is open source and available under the MIT License.

## Acknowledgments

- FER2013 Dataset creators
- TensorFlow and Keras teams
- OpenCV community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub.
