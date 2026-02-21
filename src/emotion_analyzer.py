import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # Suppress TensorFlow info & warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         # Disable oneDNN optimization logs


import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mediapipe as mp
import json
import time
import os
from datetime import datetime
from collections import deque
import threading

class EmotionAnalyzer:
    def __init__(self, model_path='emotion_recognition_model.h5'):
        """
        Initialize the emotion analyzer with the trained model
        """
        self.model = load_model(model_path)
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion labels (adjust based on your model)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Confidence and nervousness tracking
        self.emotion_history = deque(maxlen=50)  # Store last 50 predictions
        self.blink_history = deque(maxlen=100)   # Store blink data
        self.movement_history = deque(maxlen=30) # Store head movement data
        
        # Initialize face landmarks for feature extraction
        self.previous_landmarks = None
        self.blink_counter = 0
        self.last_blink_time = time.time()
        
        # Logging setup
        self.log_data = []
        self.last_log_time = time.time()
        self.log_interval = 15  # seconds
        
    def preprocess_face(self, face_img):
        """
        Preprocess face image for emotion recognition model
        """ 
        # Resize to model input size (usually 48x48 for emotion recognition)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        """
        # Vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_facial_features(self, landmarks, image_shape):
        """
        Extract facial features for confidence and nervousness analysis
        """
        features = {}
        
        if landmarks is None:
            return features
        
        # Convert landmarks to numpy array
        landmarks_array = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] 
                                   for lm in landmarks.landmark])
        
        # Eye landmarks (approximate indices for MediaPipe face mesh)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Extract eye regions
        left_eye = landmarks_array[left_eye_indices]
        right_eye = landmarks_array[right_eye_indices]
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_eye_aspect_ratio(left_eye[:6])  # Use first 6 points
        right_ear = self.calculate_eye_aspect_ratio(right_eye[:6])
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink detection
        if avg_ear < 0.25:  # Threshold for blink detection
            self.blink_counter += 1
        
        # Calculate blink rate
        current_time = time.time()
        self.blink_history.append(current_time)
        recent_blinks = sum(1 for t in self.blink_history if current_time - t < 60)  # Blinks per minute
        
        # Head pose estimation (simplified)
        nose_tip = landmarks_array[1]  # Nose tip
        chin = landmarks_array[18]     # Chin
        left_face = landmarks_array[234]  # Left face
        right_face = landmarks_array[454] # Right face
        
        # Calculate head movement
        if self.previous_landmarks is not None:
            movement = np.linalg.norm(nose_tip - self.previous_landmarks)
            self.movement_history.append(movement)
        
        self.previous_landmarks = nose_tip.copy()
        
        # Calculate movement variance (indicator of nervousness)
        movement_variance = np.var(list(self.movement_history)) if len(self.movement_history) > 5 else 0
        
        features = {
            'eye_aspect_ratio': avg_ear,
            'blink_rate': recent_blinks,
            'head_movement_variance': movement_variance,
            'facial_symmetry': self.calculate_facial_symmetry(landmarks_array),
            'mouth_openness': self.calculate_mouth_openness(landmarks_array)
        }
        
        return features
    
    def calculate_facial_symmetry(self, landmarks):
        """
        Calculate facial symmetry score
        """
        # Use key facial points for symmetry calculation
        left_eye = landmarks[33]
        right_eye = landmarks[362]
        nose_tip = landmarks[1]
        
        # Calculate distances from nose to eyes
        left_distance = np.linalg.norm(nose_tip - left_eye)
        right_distance = np.linalg.norm(nose_tip - right_eye)
        
        # Symmetry score (closer to 1 means more symmetric)
        symmetry = min(left_distance, right_distance) / max(left_distance, right_distance)
        return symmetry
    
    def calculate_mouth_openness(self, landmarks):
        """
        Calculate mouth openness ratio
        """
        # Mouth landmarks (approximate)
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Vertical mouth distance
        vertical_dist = np.linalg.norm(upper_lip - lower_lip)
        # Horizontal mouth distance
        horizontal_dist = np.linalg.norm(left_mouth - right_mouth)
        
        # Mouth openness ratio
        openness = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
        return openness
    
    def analyze_confidence_nervousness(self, emotion_probs, facial_features):
        """
        Analyze confidence and nervousness based on emotions and facial features
        """
        # Confidence indicators
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['fear', 'sad', 'angry']
        
        positive_score = sum([emotion_probs[i] for i, label in enumerate(self.emotion_labels) 
                             if label in positive_emotions])
        negative_score = sum([emotion_probs[i] for i, label in enumerate(self.emotion_labels) 
                             if label in negative_emotions])
        
        # Base confidence from emotion
        emotion_confidence = positive_score - negative_score * 0.5
        
        # Adjust confidence based on facial features
        feature_confidence = 0
        if facial_features:
            # High blink rate indicates nervousness
            if facial_features['blink_rate'] > 20:  # More than 20 blinks per minute
                feature_confidence -= 0.2
            elif facial_features['blink_rate'] < 10:  # Very low blink rate
                feature_confidence -= 0.1
            
            # High movement variance indicates nervousness
            if facial_features['head_movement_variance'] > 50:
                feature_confidence -= 0.3
            
            # Facial symmetry affects confidence perception
            feature_confidence += (facial_features['facial_symmetry'] - 0.8) * 0.5
        
        overall_confidence = max(0, min(1, 0.5 + emotion_confidence + feature_confidence))
        
        # Nervousness calculation
        nervousness_score = 0
        if facial_features:
            # High blink rate
            if facial_features['blink_rate'] > 25:
                nervousness_score += 0.4
            
            # High movement variance
            if facial_features['head_movement_variance'] > 100:
                nervousness_score += 0.4
            
            # Fear emotion increases nervousness
            fear_index = self.emotion_labels.index('fear') if 'fear' in self.emotion_labels else -1
            if fear_index >= 0:
                nervousness_score += emotion_probs[fear_index] * 0.3
        
        nervousness_score = min(1, nervousness_score)
        
        return overall_confidence, nervousness_score
    
    def process_frame(self, frame):
        """
        Process a single frame for emotion recognition and feature extraction
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'face_detected': False,
            'emotions': {},
            'confidence_score': 0,
            'nervousness_score': 0,
            'facial_features': {}
        }
        
        if results.detections:
            for detection in results.detections:
                # Extract face bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                
                # Extract face region
                face_region = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                
                if face_region.size > 0:
                    # Preprocess face for emotion recognition
                    processed_face = self.preprocess_face(face_region)
                    
                    # Predict emotions
                    emotion_probs = self.model.predict(processed_face, verbose=0)[0]
                    
                    # Get facial landmarks for feature extraction
                    mesh_results = self.face_mesh.process(rgb_frame)
                    facial_features = {}
                    
                    if mesh_results.multi_face_landmarks:
                        landmarks = mesh_results.multi_face_landmarks[0]
                        facial_features = self.extract_facial_features(landmarks, frame.shape)
                    
                    # Analyze confidence and nervousness
                    confidence, nervousness = self.analyze_confidence_nervousness(emotion_probs, facial_features)
                    
                    # Store emotion history for stability
                    self.emotion_history.append(emotion_probs)
                    
                    # Create emotion dictionary
                    emotions_dict = {label: float(prob) for label, prob in zip(self.emotion_labels, emotion_probs)}
                    
                    analysis_result.update({
                        'face_detected': True,
                        'emotions': emotions_dict,
                        'dominant_emotion': self.emotion_labels[np.argmax(emotion_probs)],
                        'confidence_score': float(confidence),
                        'nervousness_score': float(nervousness),
                        'facial_features': facial_features
                    })
                    
                    break  # Process only the first detected face
        
        return analysis_result
    
    def log_analysis(self, analysis_data, log_file='emotion_analysis_log.json'):
        """
        Log analysis data to JSON file
        """
        self.log_data.append(analysis_data)
        
        # Write to file every log_interval seconds
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            try:
                # Load existing data if file exists
                existing_data = []
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        existing_data = json.load(f)
                
                # Append new data
                existing_data.extend(self.log_data)
                
                # Write back to file
                with open(log_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
                print(f"Logged {len(self.log_data)} entries to {log_file}")
                self.log_data = []  # Clear the buffer
                self.last_log_time = current_time
                
            except Exception as e:
                print(f"Error writing to log file: {e}")
    
    def get_analysis_summary(self):
        """
        Get summary statistics from recent analysis
        """
        if len(self.emotion_history) < 5:
            return None
        
        recent_emotions = list(self.emotion_history)[-10:]  # Last 10 predictions
        avg_emotions = np.mean(recent_emotions, axis=0)
        
        summary = {
            'average_emotions': {label: float(prob) for label, prob in zip(self.emotion_labels, avg_emotions)},
            'dominant_emotion': self.emotion_labels[np.argmax(avg_emotions)],
            'emotion_stability': float(1 - np.std(recent_emotions)),
            'total_frames_processed': len(self.emotion_history)
        }
        
        return summary
