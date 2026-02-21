import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime
import json

class EyeTracker:
    def __init__(self, log_file='eye_tracking_log.json'):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.log_file = log_file
        self.eye_tracking_data = []
        self.start_time = None
        
        # Gaze metrics
        self.gaze_patterns = {
            'looking_at_camera': 0,
            'looking_away': 0,
            'looking_up': 0,
            'looking_down': 0,
            'looking_left': 0,
            'looking_right': 0
        }
        self.total_frames = 0
        
    def _get_head_pose(self, frame, landmarks):
        img_h, img_w, _ = frame.shape
        face_3d = []
        face_2d = []

        # Key landmarks for head pose estimation
        # Nose tip: 1, Chin: 152, Left eye left corner: 33, Right eye right corner: 263, Left Mouth corner: 61, Right Mouth corner: 291
        key_landmarks = [1, 152, 33, 263, 61, 291]

        for idx, lm in enumerate(landmarks.landmark):
            if idx in key_landmarks:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])       
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])

        # Distortion matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        # cv2.RQDecomp3x3 returns 6 values: angles, mtxR, mtxQ, Qx, Qy, Qz
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree (Yaw)
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        return x, y, z

    def _analyze_gaze(self, frame, face_landmarks):
        pitch, yaw, roll = self._get_head_pose(frame, face_landmarks)
        
        # Determine gaze direction based on angles
        # Adjust thresholds as needed
        direction = "center"
        is_looking_at_camera = True
        
        if pitch < -10:
            direction = "down"
            self.gaze_patterns['looking_down'] += 1
            is_looking_at_camera = False
        elif pitch > 10:
            direction = "up"
            self.gaze_patterns['looking_up'] += 1
            is_looking_at_camera = False
        elif yaw < -10:
            direction = "left"
            self.gaze_patterns['looking_left'] += 1
            is_looking_at_camera = False
        elif yaw > 10:
            direction = "right"
            self.gaze_patterns['looking_right'] += 1
            is_looking_at_camera = False
        else:
            direction = "center"
            self.gaze_patterns['looking_at_camera'] += 1
            is_looking_at_camera = True
            
        if not is_looking_at_camera:
            self.gaze_patterns['looking_away'] += 1
            
        self.total_frames += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'direction': direction,
            'is_looking_at_camera': is_looking_at_camera
        }
    
    def analyze_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            analysis = self._analyze_gaze(frame, face_landmarks)
            self.eye_tracking_data.append(analysis)
            return analysis
        return None
    
    def get_analysis_summary(self):
        if not self.eye_tracking_data:
            return None
            
        total_frames = self.total_frames if self.total_frames > 0 else 1
        
        summary = {
            'camera_focus_percentage': (self.gaze_patterns['looking_at_camera'] / total_frames) * 100,
            'distraction_percentage': (self.gaze_patterns['looking_away'] / total_frames) * 100,
            'looking_up_percentage': (self.gaze_patterns['looking_up'] / total_frames) * 100,
            'looking_down_percentage': (self.gaze_patterns['looking_down'] / total_frames) * 100,
            'looking_left_percentage': (self.gaze_patterns['looking_left'] / total_frames) * 100,
            'looking_right_percentage': (self.gaze_patterns['looking_right'] / total_frames) * 100,
            'total_frames': total_frames
        }
        
        return summary
    
    def save_log(self):
        if self.eye_tracking_data:
            with open(self.log_file, 'w') as f:
                json.dump({
                    'eye_tracking_data': self.eye_tracking_data,
                    'summary': self.get_analysis_summary()
                }, f, indent=4)
    
    def reset(self):
        self.eye_tracking_data = []
        self.gaze_patterns = {
            'looking_at_camera': 0,
            'looking_away': 0,
            'looking_up': 0,
            'looking_down': 0,
            'looking_left': 0,
            'looking_right': 0
        }
        self.total_frames = 0