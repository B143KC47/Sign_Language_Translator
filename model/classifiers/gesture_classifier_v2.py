import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import os
from collections import deque


class ImprovedGestureClassifier:
    def __init__(self, num_classes=26, sequence_length=30):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.model = None
        self.sequence_model = None
        self.gesture_sequence = deque(maxlen=sequence_length)
        
        self.gesture_labels = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
            16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
            23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE', 27: 'DELETE'
        }
        
        self.word_gestures = {
            'HELLO': [7, 4, 11, 11, 14],
            'THANKS': [19, 7, 0, 13, 10, 18],
            'YES': [24, 4, 18],
            'NO': [13, 14],
            'PLEASE': [15, 11, 4, 0, 18, 4],
            'SORRY': [18, 14, 17, 17, 24]
        }
        
    def build_static_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.num_classes + 2, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_sequence_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, 
                            input_shape=(self.sequence_length, input_shape)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(self.word_gestures), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.sequence_model = model
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.build_static_model(X.shape[1])
        
        y_categorical = keras.utils.to_categorical(y, self.num_classes + 2)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict_with_confidence_threshold(self, features, threshold=0.5):
        if self.model is None:
            return None, 0
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        predictions = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if confidence < threshold:
            return None, confidence
        
        gesture = self.gesture_labels.get(predicted_class, '?')
        return gesture, confidence
    
    def update_sequence(self, features):
        self.gesture_sequence.append(features)
    
    def detect_word_gesture(self):
        if len(self.gesture_sequence) < 10:
            return None
        
        recent_gestures = list(self.gesture_sequence)[-20:]
        
        for word, pattern in self.word_gestures.items():
            if self._match_pattern(recent_gestures, pattern):
                return word
        
        return None
    
    def _match_pattern(self, sequence, pattern):
        pattern_features = []
        for gesture_id in pattern:
            if gesture_id < len(self.gesture_labels):
                pattern_features.append(gesture_id)
        
        return False
    
    def save_models(self, static_path='gesture_model_v2.h5', 
                   sequence_path='sequence_model_v2.h5'):
        if self.model:
            self.model.save(static_path)
        if self.sequence_model:
            self.sequence_model.save(sequence_path)
    
    def load_models(self, static_path='gesture_model_v2.h5',
                   sequence_path='sequence_model_v2.h5'):
        if os.path.exists(static_path):
            self.model = keras.models.load_model(static_path)
        if os.path.exists(sequence_path):
            self.sequence_model = keras.models.load_model(sequence_path)
        return self.model is not None