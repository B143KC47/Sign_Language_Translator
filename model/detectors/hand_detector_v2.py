import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class ImprovedHandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.history = deque(maxlen=10)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
    def preprocess_image(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_hands(self, image, preprocess=True):
        if preprocess:
            processed = self.preprocess_image(image)
        else:
            processed = image
            
        image_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results
    
    def get_landmarks(self, results, image_shape):
        all_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = image_shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append([x, y, landmark.z])
                all_landmarks.append(np.array(landmarks))
        return all_landmarks
    
    def extract_advanced_features(self, landmarks):
        if len(landmarks) == 0:
            return None
        
        landmarks = landmarks[0]
        features = []
        
        wrist = landmarks[0]
        landmarks_normalized = landmarks - wrist
        
        max_dist = np.max(np.linalg.norm(landmarks_normalized, axis=1))
        if max_dist > 0:
            landmarks_normalized = landmarks_normalized / max_dist
        
        features.extend(landmarks_normalized[:, :2].flatten())
        
        for i, j in self.hand_connections:
            vec = landmarks_normalized[j] - landmarks_normalized[i]
            angle = np.arctan2(vec[1], vec[0])
            distance = np.linalg.norm(vec[:2])
            features.extend([angle, distance])
        
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]
        
        for tip, base in zip(finger_tips, finger_bases):
            flexion = np.linalg.norm(landmarks_normalized[tip][:2] - landmarks_normalized[base][:2])
            features.append(flexion)
        
        palm_center = np.mean(landmarks_normalized[[0, 5, 9, 13, 17]], axis=0)
        palm_size = np.mean([np.linalg.norm(landmarks_normalized[i][:2] - palm_center[:2]) 
                            for i in [5, 9, 13, 17]])
        features.append(palm_size)
        
        centroid = np.mean(landmarks_normalized[:, :2], axis=0)
        features.extend(centroid)
        
        spread = np.std(landmarks_normalized[:, :2])
        features.append(spread)
        
        return np.array(features)
    
    def smooth_landmarks(self, landmarks):
        if landmarks:
            self.history.append(landmarks[0])
            if len(self.history) > 3:
                smoothed = np.mean(list(self.history)[-3:], axis=0)
                return [smoothed.astype(int)]
        return landmarks
    
    def draw_enhanced_skeleton(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]
                    
                    h, w, _ = image.shape
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    cv2.line(image, start_point, end_point, (0, 255, 0), 3)
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    if i in [4, 8, 12, 16, 20]:
                        cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)
                    elif i == 0:
                        cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
                    else:
                        cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)
        
        return image
    
    def get_hand_orientation(self, landmarks):
        if len(landmarks) == 0:
            return None
        
        landmarks = landmarks[0]
        wrist = landmarks[0]
        middle_base = landmarks[9]
        
        direction = middle_base - wrist
        angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return angle