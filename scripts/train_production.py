import argparse
import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from datetime import datetime
import numpy as np
from typing import Dict, Tuple

from core.data_pipeline import SignLanguageDataset
from core.models import SignLanguageTransformer
from core.model_versioning import ModelRegistry, ModelLifecycleManager
from core.monitoring import ProductionMonitor


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_datasets(language: str, config: Dict) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    dataset = SignLanguageDataset(language, config['data']['path'])
    
    vocab = dataset.load_vocabulary()
    print(f"Loaded vocabulary with {len(vocab)} signs for {language}")
    
    train_data = dataset.create_tf_dataset(
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_data = dataset.create_tf_dataset(
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    test_data = dataset.create_tf_dataset(
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    total_samples = sum(1 for _ in train_data.unbatch())
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    train_data = train_data.take(train_size // config['training']['batch_size'])
    val_data = val_data.skip(train_size // config['training']['batch_size']).take(val_size // config['training']['batch_size'])
    test_data = test_data.skip((train_size + val_size) // config['training']['batch_size'])
    
    return train_data, val_data, test_data


def create_callbacks(model_name: str, language: str, config: Dict) -> list:
    callbacks = []
    
    log_dir = os.path.join(config['training']['log_dir'], 
                          f"{model_name}_{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch='10, 20'
    ))
    
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(log_dir, 'checkpoint_{epoch:02d}_{val_accuracy:.4f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    ))
    
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['training'].get('early_stopping_patience', 10),
        restore_best_weights=True,
        verbose=1
    ))
    
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ))
    
    class MetricsLogger(tf.keras.callbacks.Callback):
        def __init__(self, monitor):
            self.monitor = monitor
            
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                self.monitor.log_translation_quality(
                    language=language,
                    accuracy=logs.get('val_accuracy', 0)
                )
    
    if 'monitor' in config:
        callbacks.append(MetricsLogger(ProductionMonitor(config['monitor'])))
    
    return callbacks


def train_model(language: str, config: Dict):
    print(f"\n=== Training Sign Language Model for {language} ===\n")
    
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    train_data, val_data, test_data = prepare_datasets(language, config)
    
    lang_config = next(
        (lang for lang in config['languages']['sign_languages'] 
         if lang['code'] == language),
        None
    )
    
    if not lang_config:
        raise ValueError(f"Language {language} not found in config")
    
    with strategy.scope():
        model = SignLanguageTransformer(
            num_classes=lang_config['vocabulary_size'],
            max_length=config['models']['recognition']['max_sequence_length'],
            d_model=config['models']['recognition']['embedding_dim'],
            num_heads=config['models']['recognition']['num_heads'],
            num_layers=config['models']['recognition']['num_layers'],
            dropout_rate=config['models']['recognition']['dropout_rate']
        )
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['training']['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name='loss')
            ]
        )
    
    model.build(input_shape=(None, None, 366))
    model.summary()
    
    callbacks = create_callbacks(f"sign_language_{language}", language, config)
    
    history = model.fit(
        train_data,
        epochs=config['training']['epochs'],
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Evaluating on Test Set ===")
    test_results = model.evaluate(test_data, verbose=1)
    
    metrics = {
        'accuracy': float(history.history['accuracy'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_results[1]),
        'top_5_accuracy': float(history.history['top_5_accuracy'][-1]),
        'val_top_5_accuracy': float(history.history['val_top_5_accuracy'][-1]),
        'final_loss': float(history.history['loss'][-1]),
        'val_loss': float(history.history['val_loss'][-1])
    }
    
    parameters = {
        'epochs': len(history.history['loss']),
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': 'adam',
        'architecture': 'transformer',
        'd_model': config['models']['recognition']['embedding_dim'],
        'num_heads': config['models']['recognition']['num_heads'],
        'num_layers': config['models']['recognition']['num_layers']
    }
    
    registry = ModelRegistry(
        storage_backend=config.get('model_storage', 'local'),
        config=config.get('registry', {})
    )
    
    model_version = registry.register_model(
        model=model,
        model_name=f"sign_language_{language}",
        language=language,
        metrics=metrics,
        parameters=parameters,
        tags=['production_training', f'v{config["app"]["version"]}']
    )
    
    print(f"\n=== Model Registered ===")
    print(f"Version ID: {model_version.version_id}")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    
    if config.get('auto_promote', False) and metrics['test_accuracy'] > config.get('min_accuracy', 0.8):
        registry.promote_model(
            model_name=f"sign_language_{language}",
            language=language,
            version_id=model_version.version_id,
            target_status='production'
        )
        print(f"\nModel automatically promoted to production!")
    
    return model_version


def main():
    parser = argparse.ArgumentParser(description='Train production sign language models')
    parser.add_argument('--config', type=str, default='config/production_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--language', type=str, required=True,
                       help='Sign language to train (e.g., ASL, BSL, JSL)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device IDs to use')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    config = load_config(args.config)
    
    tf.config.optimizer.set_jit(True)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    train_model(args.language, config)


if __name__ == "__main__":
    main()