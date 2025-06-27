import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from typing import Dict, List, Tuple, Optional
import h5py
import json
import os
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging


class SignLanguageDataset:
    def __init__(self, language_code: str, data_dir: str = "data"):
        self.language_code = language_code
        self.data_dir = os.path.join(data_dir, language_code)
        self.logger = logging.getLogger(__name__)
        
        self.metadata = {
            'ASL': {
                'classes': 1000,
                'fps': 30,
                'keypoints': 543,  # 21 hand + 33 pose + 468 face
                'dataset_urls': [
                    'https://example.com/asl_dataset_v1.zip',
                    'https://example.com/wlasl_dataset.zip'
                ]
            },
            'BSL': {
                'classes': 800,
                'fps': 25,
                'keypoints': 543,
                'dataset_urls': [
                    'https://example.com/bsl_corpus.zip'
                ]
            },
            'JSL': {
                'classes': 1200,
                'fps': 30,
                'keypoints': 543,
                'dataset_urls': [
                    'https://example.com/jsl_dataset.zip'
                ]
            },
            'CSL': {
                'classes': 1500,
                'fps': 30,
                'keypoints': 543,
                'dataset_urls': [
                    'https://example.com/csl_dataset.zip'
                ]
            }
        }
        
    def load_vocabulary(self) -> Dict[int, Dict[str, str]]:
        vocab_path = os.path.join(self.data_dir, 'vocabulary.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            return {int(k): v for k, v in vocab.items()}
        return {}
    
    def preprocess_video(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def extract_holistic_features(self, frames: np.ndarray) -> np.ndarray:
        import mediapipe as mp
        
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        all_features = []
        
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            results = holistic.process(frame_uint8)
            
            features = []
            
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
            else:
                features.extend([0] * 63)
            
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
            else:
                features.extend([0] * 63)
            
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                features.extend([0] * 132)
            
            if results.face_landmarks:
                selected_indices = [0, 13, 14, 17, 18, 78, 80, 81, 82, 87, 88, 95, 178, 181, 185]
                for idx in selected_indices:
                    if idx < len(results.face_landmarks.landmark):
                        landmark = results.face_landmarks.landmark[idx]
                        features.extend([landmark.x, landmark.y, landmark.z])
                    else:
                        features.extend([0, 0, 0])
            else:
                features.extend([0] * 45)
            
            all_features.append(features)
        
        holistic.close()
        return np.array(all_features)
    
    def create_tf_dataset(self, batch_size: int = 32, 
                         shuffle: bool = True) -> tf.data.Dataset:
        data_files = tf.io.gfile.glob(os.path.join(self.data_dir, '*.tfrecord'))
        
        dataset = tf.data.TFRecordDataset(data_files)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.map(self._parse_tfrecord, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _parse_tfrecord(self, example_proto):
        feature_description = {
            'features': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'sequence_length': tf.io.FixedLenFeature([], tf.int64),
            'language': tf.io.FixedLenFeature([], tf.string)
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        features = tf.reshape(parsed['features'], [-1, 366])
        
        return features, parsed['label']
    
    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-15, 15)
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        augmented = sequence.copy()
        
        for i in range(0, augmented.shape[1], 3):
            if i + 1 < augmented.shape[1]:
                xy = augmented[:, i:i+2]
                augmented[:, i:i+2] = np.dot(xy, rotation_matrix.T)
        
        scale = np.random.uniform(0.9, 1.1)
        augmented[:, :2] *= scale
        
        noise = np.random.normal(0, 0.01, augmented.shape)
        augmented += noise
        
        return augmented
    
    def validate_data_quality(self, features: np.ndarray) -> Tuple[bool, str]:
        if np.any(np.isnan(features)):
            return False, "Contains NaN values"
        
        if np.all(features == 0):
            return False, "All zeros - no detection"
        
        hand_features = features[:126]
        if np.sum(hand_features != 0) < 0.3 * len(hand_features):
            return False, "Insufficient hand landmarks"
        
        return True, "Valid"


class DataCollector:
    def __init__(self, storage_backend: str = "s3"):
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def collect_sample(self, features: np.ndarray, label: Optional[int], 
                      metadata: Dict) -> str:
        timestamp = datetime.utcnow().isoformat()
        
        sample_id = hashlib.sha256(
            f"{timestamp}{metadata.get('user_id', 'anonymous')}".encode()
        ).hexdigest()[:16]
        
        sample = {
            'id': sample_id,
            'timestamp': timestamp,
            'features': features.tolist(),
            'label': label,
            'metadata': metadata,
            'version': '1.0'
        }
        
        self.executor.submit(self._save_sample, sample)
        
        return sample_id
    
    def _save_sample(self, sample: Dict):
        try:
            if self.storage_backend == "s3":
                self._save_to_s3(sample)
            elif self.storage_backend == "local":
                self._save_locally(sample)
            
            self.logger.info(f"Sample {sample['id']} saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save sample {sample['id']}: {e}")
    
    def _save_to_s3(self, sample: Dict):
        pass
    
    def _save_locally(self, sample: Dict):
        date_str = datetime.utcnow().strftime("%Y%m%d")
        output_dir = f"collected_data/{date_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"{sample['id']}.json")
        with open(filepath, 'w') as f:
            json.dump(sample, f)


class ActiveLearning:
    def __init__(self, uncertainty_threshold: float = 0.3):
        self.uncertainty_threshold = uncertainty_threshold
        self.sample_pool = []
        
    def calculate_uncertainty(self, predictions: np.ndarray) -> float:
        top_2_probs = np.sort(predictions)[-2:]
        uncertainty = 1.0 - (top_2_probs[1] - top_2_probs[0])
        return uncertainty
    
    def should_collect_sample(self, predictions: np.ndarray, 
                            features: np.ndarray) -> bool:
        uncertainty = self.calculate_uncertainty(predictions)
        
        if uncertainty > self.uncertainty_threshold:
            entropy = -np.sum(predictions * np.log(predictions + 1e-10))
            
            self.sample_pool.append({
                'features': features,
                'predictions': predictions,
                'uncertainty': uncertainty,
                'entropy': entropy,
                'timestamp': datetime.utcnow()
            })
            
            return True
        
        return False
    
    def get_samples_for_labeling(self, n_samples: int = 100) -> List[Dict]:
        sorted_pool = sorted(self.sample_pool, 
                           key=lambda x: x['uncertainty'], 
                           reverse=True)
        
        selected = sorted_pool[:n_samples]
        
        self.sample_pool = sorted_pool[n_samples:]
        
        return selected