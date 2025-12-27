"""
Real-time Emotion Detection using Webcam
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import *


class EmotionDetector:
    """Real-time emotion detection from webcam"""
    
    def __init__(self, model_path=MODEL_SAVE_PATH):
        """Initialize the emotion detector"""
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_labels = EMOTIONS
        
        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 0),
            'fear': (128, 0, 128),
            'happy': (0, 255, 255),
            'sad': (255, 0, 0),
            'surprise': (255, 165, 0),
            'neutral': (128, 128, 128)
        }
    
    def preprocess_face(self, face):
        """Preprocess face image for model input"""
        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        return face
    
    def detect_emotion(self, frame):
        """Detect emotions in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            preprocessed_face = self.preprocess_face(face_roi)
            
            predictions = self.model.predict(preprocessed_face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion = self.emotion_labels[emotion_idx]
            confidence = predictions[0][emotion_idx]
            
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{emotion.capitalize()}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame
    
    def run(self):
        """Run real-time emotion detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting real-time emotion detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame = self.detect_emotion(frame)
            
            cv2.imshow('Real-time Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped")


def main():
    """Main function to run real-time detection"""
    detector = EmotionDetector()
    detector.run()


if __name__ == "__main__":
    main()
