import cv2
import numpy as np
import os
import json
from datetime import datetime


def create_enhanced_dataset(output_dir='dataset_v2', num_samples_per_class=200):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.random.seed(42)
    X = []
    y = []
    
    feature_dim = 91
    
    for class_id in range(28):
        base_pattern = np.random.randn(feature_dim) * 0.1
        
        if class_id < 5:
            base_pattern[0:20] += 0.5
        elif class_id < 10:
            base_pattern[20:40] += 0.5
        elif class_id < 15:
            base_pattern[40:60] += 0.5
        elif class_id < 20:
            base_pattern[60:80] += 0.5
        else:
            base_pattern[80:] += 0.5
        
        for _ in range(num_samples_per_class):
            features = base_pattern + np.random.randn(feature_dim) * 0.15
            
            noise_level = np.random.uniform(0, 0.1)
            features += np.random.randn(feature_dim) * noise_level
            
            rotation = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.9, 1.1)
            features[:42] *= scale
            
            X.append(features)
            y.append(class_id)
    
    X = np.array(X)
    y = np.array(y)
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    np.save(os.path.join(output_dir, 'X_enhanced.npy'), X)
    np.save(os.path.join(output_dir, 'y_enhanced.npy'), y)
    
    metadata = {
        'num_samples': len(X),
        'num_classes': 28,
        'feature_dim': feature_dim,
        'creation_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Enhanced dataset created with {len(X)} samples")
    return X, y


def load_enhanced_dataset(dataset_dir='dataset_v2'):
    X_path = os.path.join(dataset_dir, 'X_enhanced.npy')
    y_path = os.path.join(dataset_dir, 'y_enhanced.npy')
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path)
        y = np.load(y_path)
        return X, y
    return None, None


def augment_image_realtime(image):
    if np.random.random() < 0.5:
        brightness = np.random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    if np.random.random() < 0.3:
        angle = np.random.uniform(-10, 10)
        center = (image.shape[1]//2, image.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    
    return image


def draw_enhanced_text(image, texts, colors=None):
    if colors is None:
        colors = [(0, 255, 0)] * len(texts)
    
    y_offset = 30
    for i, (text, color) in enumerate(zip(texts, colors)):
        y_position = y_offset + i * 35
        
        cv2.rectangle(image, (5, y_position - 25), (len(text) * 12 + 15, y_position + 5), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (5, y_position - 25), (len(text) * 12 + 15, y_position + 5), 
                     color, 2)
        
        cv2.putText(image, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2, cv2.LINE_AA)
    
    return image


def create_visualization_window(width=1280, height=720):
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.rectangle(blank, (0, 0), (width//2, height), (40, 40, 40), -1)
    cv2.rectangle(blank, (width//2, 0), (width, height), (20, 20, 20), -1)
    
    cv2.putText(blank, "Camera Feed", (width//4 - 50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blank, "Translation", (3*width//4 - 50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return blank


class PerformanceMonitor:
    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        self.last_time = cv2.getTickCount()
        
    def update(self):
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - self.last_time) / cv2.getTickFrequency()
        self.last_time = current_time
        
        if time_elapsed > 0:
            fps = 1.0 / time_elapsed
            self.fps_history.append(fps)
        
        return time_elapsed
    
    def get_fps(self):
        if self.fps_history:
            return np.mean(list(self.fps_history))
        return 0
    
    def add_processing_time(self, time_ms):
        self.processing_times.append(time_ms)
    
    def get_avg_processing_time(self):
        if self.processing_times:
            return np.mean(list(self.processing_times))
        return 0


def save_gesture_recording(gestures, filename='recording.json'):
    data = {
        'timestamp': datetime.now().isoformat(),
        'gestures': gestures,
        'duration': len(gestures)
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filename


def load_gesture_recording(filename='recording.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['gestures']
    return []