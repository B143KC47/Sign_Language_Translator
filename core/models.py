import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class SignLanguageTransformer(keras.Model):
    def __init__(self, num_classes: int, max_length: int = 100, 
                 d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, dff: int = 1024, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_length = max_length
        self.d_model = d_model
        
        self.input_projection = keras.layers.Dense(d_model)
        
        self.positional_encoding = self.create_positional_encoding(max_length, d_model)
        
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = keras.layers.Dropout(dropout_rate)
        
        self.global_pool = keras.layers.GlobalAveragePooling1D()
        
        self.classifier = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(dff // 2, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(num_classes)
        ])
        
    def create_positional_encoding(self, max_length: int, d_model: int) -> tf.Tensor:
        position = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                          -(tf.math.log(10000.0) / d_model))
        
        pos_encoding = tf.zeros((max_length, d_model))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            [[i, j] for i in range(max_length) for j in range(0, d_model, 2)],
            tf.reshape(tf.sin(position * div_term), [-1])
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            [[i, j] for i in range(max_length) for j in range(1, d_model, 2)],
            tf.reshape(tf.cos(position * div_term), [-1])
        )
        
        return pos_encoding[tf.newaxis, ...]
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        
        x = self.input_projection(inputs)
        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, training=training)
            attention_weights.append(attn_weights)
        
        x = self.global_pool(x)
        
        outputs = self.classifier(x)
        
        return outputs, attention_weights
    
    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'd_model': self.d_model
        }


class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
    
    def call(self, x: tf.Tensor, training: bool = None) -> Tuple[tf.Tensor, tf.Tensor]:
        attn_output, attn_weights = self.mha(x, x, x, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2, attn_weights


class MultiLanguageSignModel:
    def __init__(self, config_path: str = "config/production_config.yaml"):
        self.models = {}
        self.config = self._load_config(config_path)
        self.logger = tf.get_logger()
        
    def _load_config(self, config_path: str) -> Dict:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, language_code: str) -> SignLanguageTransformer:
        if language_code in self.models:
            return self.models[language_code]
        
        model_config = next(
            (lang for lang in self.config['languages']['sign_languages'] 
             if lang['code'] == language_code), 
            None
        )
        
        if not model_config:
            raise ValueError(f"Unsupported language: {language_code}")
        
        model = SignLanguageTransformer(
            num_classes=model_config['vocabulary_size'],
            **self.config['models']['recognition']
        )
        
        model_path = model_config['model_path']
        if os.path.exists(model_path):
            model.load_weights(model_path)
            self.logger.info(f"Loaded model for {language_code}")
        else:
            self.logger.warning(f"No pre-trained model found for {language_code}")
        
        self.models[language_code] = model
        return model
    
    def predict(self, features: np.ndarray, language_code: str, 
               return_attention: bool = False) -> Dict:
        model = self.load_model(language_code)
        
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]
        
        outputs, attention_weights = model(features, training=False)
        
        probabilities = tf.nn.softmax(outputs)
        predicted_class = tf.argmax(probabilities, axis=-1)
        confidence = tf.reduce_max(probabilities, axis=-1)
        
        result = {
            'predicted_class': predicted_class.numpy()[0],
            'confidence': confidence.numpy()[0],
            'probabilities': probabilities.numpy()[0],
            'language': language_code
        }
        
        if return_attention:
            result['attention_weights'] = [w.numpy() for w in attention_weights]
        
        return result
    
    def batch_predict(self, features_list: List[np.ndarray], 
                     language_code: str) -> List[Dict]:
        model = self.load_model(language_code)
        
        max_len = max(f.shape[0] for f in features_list)
        batch_size = len(features_list)
        feature_dim = features_list[0].shape[1]
        
        padded_batch = np.zeros((batch_size, max_len, feature_dim))
        mask = np.zeros((batch_size, max_len))
        
        for i, features in enumerate(features_list):
            seq_len = features.shape[0]
            padded_batch[i, :seq_len] = features
            mask[i, :seq_len] = 1
        
        outputs, _ = model(padded_batch, training=False)
        probabilities = tf.nn.softmax(outputs)
        
        results = []
        for i in range(batch_size):
            result = {
                'predicted_class': tf.argmax(probabilities[i]).numpy(),
                'confidence': tf.reduce_max(probabilities[i]).numpy(),
                'probabilities': probabilities[i].numpy()
            }
            results.append(result)
        
        return results


class EnsembleModel:
    def __init__(self, model_configs: List[Dict]):
        self.models = []
        self.weights = []
        
        for config in model_configs:
            model = SignLanguageTransformer(**config['params'])
            model.load_weights(config['path'])
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
        
        self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def predict(self, features: np.ndarray) -> Dict:
        all_predictions = []
        
        for model, weight in zip(self.models, self.weights):
            outputs, _ = model(features, training=False)
            probabilities = tf.nn.softmax(outputs)
            all_predictions.append(probabilities * weight)
        
        ensemble_probs = tf.reduce_sum(all_predictions, axis=0)
        predicted_class = tf.argmax(ensemble_probs, axis=-1)
        confidence = tf.reduce_max(ensemble_probs, axis=-1)
        
        return {
            'predicted_class': predicted_class.numpy()[0],
            'confidence': confidence.numpy()[0],
            'probabilities': ensemble_probs.numpy()[0]
        }


class ModelOptimizer:
    @staticmethod
    def quantize_model(model: SignLanguageTransformer, 
                      dataset: tf.data.Dataset) -> tf.lite.Interpreter:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ModelOptimizer._representative_dataset(dataset)
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        return interpreter
    
    @staticmethod
    def _representative_dataset(dataset):
        for data in dataset.take(100):
            yield [tf.cast(data[0], tf.float32)]
    
    @staticmethod
    def prune_model(model: SignLanguageTransformer, 
                   sparsity: float = 0.5) -> SignLanguageTransformer:
        import tensorflow_model_optimization as tfmot
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params
        )
        
        return pruned_model