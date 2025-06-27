import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import os


class GestureClassifier:
    def __init__(self, num_classes=26):
        self.num_classes = num_classes
        self.model = None
        self.gesture_labels = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
            16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
            23: 'X', 24: 'Y', 25: 'Z'
        }
        
    def build_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.build_model(X.shape[1])
        
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history
    
    def predict(self, features):
        if self.model is None:
            return None
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        predictions = self.model.predict(features)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return self.gesture_labels.get(predicted_class, '?'), confidence
    
    def save_model(self, filepath):
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            return True
        return False