# Quick Start Guide

## 🚀 Getting Started

### 1. Extract the Repository
```bash
unzip emotion-detection-repo.zip
cd emotion-detection-repo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
Place your emotion dataset in the `data/` directory:
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
    └── (same structure)
```

### 4. Update Configuration
Edit `src/config.py` and update the `DATASET_PATH` to point to your data directory:
```python
DATASET_PATH = 'data'  # or your custom path
```

## 🎯 Usage

### Train the Model
```bash
python main.py train --epochs 50 --learning-rate 0.001
```

Or directly:
```bash
python -m src.train
```

### Evaluate the Model
```bash
python main.py evaluate
```

Or:
```bash
python -m src.evaluate
```

### Run Real-time Detection
```bash
python main.py detect
```

Or:
```bash
python -m src.real_time_detection
```

## 📁 Repository Structure

```
emotion-detection-repo/
├── src/                           # Source code
│   ├── config.py                  # Configuration
│   ├── data_loader.py             # Data preprocessing
│   ├── model.py                   # CNN architecture
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── real_time_detection.py    # Webcam detection
├── models/                        # Trained models (will be created)
├── data/                          # Your dataset goes here
├── notebooks/                     # Jupyter notebooks
│   └── Real_Time_Emotion_Detection__Basic_.ipynb
├── main.py                        # Main entry point
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── README.md                      # Full documentation
└── LICENSE                        # MIT License
```

## 🔧 Customization

### Change Image Size
Edit `src/config.py`:
```python
IMG_HEIGHT = 64  # default is 48
IMG_WIDTH = 64
```

### Change Batch Size
```python
BATCH_SIZE = 32  # default is 64
```

### Add More Emotions
```python
EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'your_emotion']
```

## 📊 Expected Results

- **Training**: Takes 30-60 minutes depending on hardware
- **Accuracy**: 65-75% on FER2013 dataset
- **Real-time**: 15-30 FPS on webcam

## 🐛 Troubleshooting

### CUDA/GPU Issues
If you encounter GPU issues:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU only
```

### Webcam Not Opening
- Check camera permissions
- Ensure no other app is using the webcam
- Try different camera index: `cv2.VideoCapture(1)` instead of `0`

### Import Errors
Make sure you're in the project directory and all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## 📝 Notes

- The original Jupyter notebook is preserved in `notebooks/`
- Models are saved in `models/` directory
- All commented code has been removed for cleaner production code
- Code is organized into logical modules for better maintainability

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Support

For issues or questions, please open an issue on GitHub.

Happy coding! 🎉
