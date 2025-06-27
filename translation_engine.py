import numpy as np
from collections import deque
import time


class TranslationEngine:
    def __init__(self, buffer_size=10, gesture_timeout=2.0):
        self.gesture_buffer = deque(maxlen=buffer_size)
        self.word_buffer = []
        self.current_word = ""
        self.translation = ""
        self.last_gesture_time = time.time()
        self.gesture_timeout = gesture_timeout
        self.min_gesture_duration = 0.3
        self.gesture_start_time = None
        self.current_gesture = None
        self.confidence_history = deque(maxlen=5)
        
        self.common_words = {
            'HELLO': 'Hello',
            'THANKS': 'Thanks',
            'YES': 'Yes',
            'NO': 'No',
            'PLEASE': 'Please',
            'SORRY': 'Sorry',
            'HELP': 'Help',
            'STOP': 'Stop',
            'GO': 'Go',
            'COME': 'Come'
        }
        
        self.gesture_transitions = {
            ('A', 'A'): 0.8,
            ('B', 'B'): 0.8,
            ('C', 'C'): 0.8,
        }
        
    def add_gesture(self, gesture, confidence):
        current_time = time.time()
        
        if gesture is None or confidence < 0.6:
            return
        
        self.confidence_history.append(confidence)
        avg_confidence = np.mean(list(self.confidence_history))
        
        if avg_confidence < 0.7:
            return
        
        if self.current_gesture != gesture:
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            return
        
        gesture_duration = current_time - self.gesture_start_time
        
        if gesture_duration < self.min_gesture_duration:
            return
        
        if current_time - self.last_gesture_time > self.gesture_timeout:
            if self.current_word:
                self.word_buffer.append(self.current_word)
                self.current_word = ""
        
        self.gesture_buffer.append((gesture, confidence, current_time))
        
        if self._is_gesture_stable():
            if gesture == 'SPACE':
                if self.current_word:
                    self.word_buffer.append(self.current_word)
                    self.current_word = ""
            elif gesture == 'DELETE':
                if self.current_word:
                    self.current_word = self.current_word[:-1]
                elif self.word_buffer:
                    self.word_buffer.pop()
            else:
                self.current_word += gesture
        
        self.last_gesture_time = current_time
    
    def _is_gesture_stable(self):
        if len(self.gesture_buffer) < 3:
            return False
        
        recent = list(self.gesture_buffer)[-3:]
        gestures = [g[0] for g in recent]
        confidences = [g[1] for g in recent]
        
        if len(set(gestures)) == 1 and min(confidences) > 0.7:
            return True
        
        return False
    
    def detect_word_completion(self):
        if not self.current_word:
            return None
        
        current_time = time.time()
        if current_time - self.last_gesture_time > 1.0:
            word = self.current_word
            self.word_buffer.append(word)
            self.current_word = ""
            return word
        
        return None
    
    def get_translation(self):
        translation_parts = []
        
        for word in self.word_buffer:
            if word in self.common_words:
                translation_parts.append(self.common_words[word])
            else:
                translation_parts.append(word.lower())
        
        if self.current_word:
            translation_parts.append(f"[{self.current_word}]")
        
        return ' '.join(translation_parts)
    
    def clear(self):
        self.gesture_buffer.clear()
        self.word_buffer.clear()
        self.current_word = ""
        self.confidence_history.clear()
        self.current_gesture = None
    
    def get_statistics(self):
        stats = {
            'words_completed': len(self.word_buffer),
            'current_word_length': len(self.current_word),
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0,
            'gesture_rate': len(self.gesture_buffer) / self.buffer_size if self.gesture_buffer else 0
        }
        return stats


class GestureTransitionDetector:
    def __init__(self, transition_threshold=0.3):
        self.previous_features = None
        self.transition_threshold = transition_threshold
        self.transition_history = deque(maxlen=10)
        
    def detect_transition(self, features):
        if self.previous_features is None:
            self.previous_features = features
            return False
        
        distance = np.linalg.norm(features - self.previous_features)
        
        self.transition_history.append(distance)
        
        is_transition = distance > self.transition_threshold
        
        if len(self.transition_history) >= 3:
            recent_distances = list(self.transition_history)[-3:]
            if all(d > self.transition_threshold for d in recent_distances):
                is_transition = True
        
        self.previous_features = features.copy()
        
        return is_transition
    
    def reset(self):
        self.previous_features = None
        self.transition_history.clear()