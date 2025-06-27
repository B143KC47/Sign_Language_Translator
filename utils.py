import cv2
import numpy as np
import os
import json


def create_sample_dataset(output_dir='dataset'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.random.seed(42)
    X = []
    y = []
    
    for class_id in range(26):
        for _ in range(100):
            features = np.random.randn(63) * 0.3
            
            if class_id < 5:
                features[0:15] += np.random.randn(15) * 0.1
            elif class_id < 10:
                features[15:30] += np.random.randn(15) * 0.1
            elif class_id < 15:
                features[30:45] += np.random.randn(15) * 0.1
            else:
                features[45:60] += np.random.randn(15) * 0.1
            
            X.append(features)
            y.append(class_id)
    
    X = np.array(X)
    y = np.array(y)
    
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    print(f"Sample dataset created with {len(X)} samples")
    return X, y


def load_dataset(dataset_dir='dataset'):
    X_path = os.path.join(dataset_dir, 'X.npy')
    y_path = os.path.join(dataset_dir, 'y.npy')
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path)
        y = np.load(y_path)
        return X, y
    return None, None


def preprocess_image(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image


def draw_text(image, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA)
    return image


class TranslationBuffer:
    def __init__(self, buffer_size=5):
        self.buffer = []
        self.buffer_size = buffer_size
        self.translation = ""
        
    def add_gesture(self, gesture, confidence):
        if confidence > 0.7:
            self.buffer.append(gesture)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
            
            if len(self.buffer) >= 3:
                most_common = max(set(self.buffer[-3:]), key=self.buffer[-3:].count)
                if self.buffer[-3:].count(most_common) >= 2:
                    if not self.translation or self.translation[-1] != most_common:
                        self.translation += most_common
    
    def get_translation(self):
        return self.translation
    
    def clear(self):
        self.translation = ""
        self.buffer = []