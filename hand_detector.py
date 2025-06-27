import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    
    def draw_skeleton(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        return image
    
    def extract_hand_features(self, landmarks):
        if len(landmarks) == 0:
            return None
        
        landmarks = landmarks[0]
        
        base_x = landmarks[0, 0]
        base_y = landmarks[0, 1]
        
        relative_landmarks = landmarks.copy()
        relative_landmarks[:, 0] -= base_x
        relative_landmarks[:, 1] -= base_y
        
        max_dist = np.max(np.sqrt(relative_landmarks[:, 0]**2 + relative_landmarks[:, 1]**2))
        if max_dist > 0:
            relative_landmarks[:, 0] /= max_dist
            relative_landmarks[:, 1] /= max_dist
        
        return relative_landmarks.flatten()