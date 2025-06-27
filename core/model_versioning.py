import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
import boto3
import mlflow
import mlflow.tensorflow
from packaging import version
import tensorflow as tf
from abc import ABC, abstractmethod
import logging


@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    language: str
    created_at: datetime
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    status: str  # 'staging', 'production', 'archived'
    path: str
    checksum: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    def __init__(self, storage_backend: str = "s3", config: Dict = None):
        self.storage_backend = storage_backend
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if storage_backend == "s3":
            self.s3_client = boto3.client('s3')
            self.bucket_name = config.get('s3_bucket', 'sign-language-models')
        
        self.local_cache_dir = config.get('local_cache', '/tmp/model_cache')
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'sqlite:///mlflow.db'))
        
    def register_model(self, model: tf.keras.Model, 
                      model_name: str,
                      language: str,
                      metrics: Dict[str, float],
                      parameters: Dict[str, Any],
                      tags: List[str] = None) -> ModelVersion:
        version_id = self._generate_version_id(model_name, language)
        
        temp_path = os.path.join(self.local_cache_dir, f"{version_id}_temp")
        model.save(temp_path)
        
        checksum = self._calculate_checksum(temp_path)
        
        if self.storage_backend == "s3":
            s3_path = f"{model_name}/{language}/{version_id}/model"
            self._upload_to_s3(temp_path, s3_path)
            final_path = f"s3://{self.bucket_name}/{s3_path}"
        else:
            final_path = os.path.join(self.local_cache_dir, model_name, language, version_id)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.move(temp_path, final_path)
        
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            language=language,
            created_at=datetime.utcnow(),
            metrics=metrics,
            parameters=parameters,
            tags=tags or [],
            status='staging',
            path=final_path,
            checksum=checksum
        )
        
        self._save_metadata(model_version)
        
        with mlflow.start_run():
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            mlflow.tensorflow.log_model(model, model_name)
            mlflow.set_tags({
                'language': language,
                'version_id': version_id,
                'checksum': checksum
            })
        
        self.logger.info(f"Registered model {model_name} version {version_id}")
        
        return model_version
    
    def get_model(self, model_name: str, language: str, 
                 version_id: Optional[str] = None) -> Tuple[tf.keras.Model, ModelVersion]:
        if version_id is None:
            version_id = self._get_latest_production_version(model_name, language)
        
        metadata = self._load_metadata(model_name, language, version_id)
        
        local_path = os.path.join(self.local_cache_dir, model_name, language, version_id)
        
        if not os.path.exists(local_path):
            if self.storage_backend == "s3":
                self._download_from_s3(metadata.path, local_path)
            else:
                raise FileNotFoundError(f"Model not found: {local_path}")
        
        if not self._verify_checksum(local_path, metadata.checksum):
            raise ValueError("Model checksum verification failed")
        
        model = tf.keras.models.load_model(local_path)
        
        return model, metadata
    
    def promote_model(self, model_name: str, language: str, 
                     version_id: str, target_status: str = 'production'):
        metadata = self._load_metadata(model_name, language, version_id)
        
        if target_status == 'production':
            current_prod = self._get_production_versions(model_name, language)
            for prod_version in current_prod:
                prod_version.status = 'archived'
                self._save_metadata(prod_version)
        
        metadata.status = target_status
        self._save_metadata(metadata)
        
        self.logger.info(f"Promoted {model_name} {version_id} to {target_status}")
    
    def list_versions(self, model_name: str, language: str) -> List[ModelVersion]:
        metadata_dir = os.path.join(self.local_cache_dir, 'metadata', model_name, language)
        
        if not os.path.exists(metadata_dir):
            return []
        
        versions = []
        for filename in os.listdir(metadata_dir):
            if filename.endswith('.json'):
                with open(os.path.join(metadata_dir, filename), 'r') as f:
                    versions.append(ModelVersion.from_dict(json.load(f)))
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def _generate_version_id(self, model_name: str, language: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_str = f"{model_name}_{language}_{timestamp}_{np.random.randint(1000)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]
    
    def _calculate_checksum(self, model_path: str) -> str:
        sha256_hash = hashlib.sha256()
        
        for root, dirs, files in os.walk(model_path):
            for file in sorted(files):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _verify_checksum(self, model_path: str, expected_checksum: str) -> bool:
        actual_checksum = self._calculate_checksum(model_path)
        return actual_checksum == expected_checksum
    
    def _save_metadata(self, model_version: ModelVersion):
        metadata_dir = os.path.join(
            self.local_cache_dir, 'metadata', 
            model_version.model_name, model_version.language
        )
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_path = os.path.join(metadata_dir, f"{model_version.version_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)
    
    def _load_metadata(self, model_name: str, language: str, version_id: str) -> ModelVersion:
        metadata_path = os.path.join(
            self.local_cache_dir, 'metadata', 
            model_name, language, f"{version_id}.json"
        )
        
        with open(metadata_path, 'r') as f:
            return ModelVersion.from_dict(json.load(f))
    
    def _get_latest_production_version(self, model_name: str, language: str) -> str:
        versions = self.list_versions(model_name, language)
        prod_versions = [v for v in versions if v.status == 'production']
        
        if not prod_versions:
            raise ValueError(f"No production version found for {model_name} {language}")
        
        return prod_versions[0].version_id
    
    def _get_production_versions(self, model_name: str, language: str) -> List[ModelVersion]:
        versions = self.list_versions(model_name, language)
        return [v for v in versions if v.status == 'production']
    
    def _upload_to_s3(self, local_path: str, s3_path: str):
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_path)
                s3_key = os.path.join(s3_path, relative_path).replace('\\', '/')
                
                self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
    
    def _download_from_s3(self, s3_path: str, local_path: str):
        s3_prefix = s3_path.replace(f"s3://{self.bucket_name}/", "")
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_file_path = os.path.join(local_path, relative_path)
                    
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)


class ABTestManager:
    def __init__(self, registry: ModelRegistry, config: Dict):
        self.registry = registry
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_tests = {}
        
    def create_test(self, test_name: str, model_name: str, language: str,
                   variant_a: str, variant_b: str, 
                   traffic_split: float = 0.5) -> Dict:
        test_config = {
            'test_name': test_name,
            'model_name': model_name,
            'language': language,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split,
            'created_at': datetime.utcnow(),
            'metrics': {
                'variant_a': {'requests': 0, 'accuracy': [], 'latency': []},
                'variant_b': {'requests': 0, 'accuracy': [], 'latency': []}
            }
        }
        
        self.active_tests[test_name] = test_config
        
        self.logger.info(f"Created A/B test {test_name}")
        
        return test_config
    
    def get_variant(self, test_name: str) -> Tuple[str, str]:
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        test = self.active_tests[test_name]
        
        if np.random.random() < test['traffic_split']:
            return 'variant_a', test['variant_a']
        else:
            return 'variant_b', test['variant_b']
    
    def record_metric(self, test_name: str, variant: str, 
                     metric_name: str, value: float):
        if test_name not in self.active_tests:
            return
        
        test = self.active_tests[test_name]
        
        if variant in test['metrics']:
            if metric_name == 'requests':
                test['metrics'][variant]['requests'] += 1
            elif metric_name in test['metrics'][variant]:
                test['metrics'][variant][metric_name].append(value)
    
    def get_test_results(self, test_name: str) -> Dict:
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        test = self.active_tests[test_name]
        results = {
            'test_name': test_name,
            'duration': (datetime.utcnow() - test['created_at']).total_seconds(),
            'variants': {}
        }
        
        for variant in ['variant_a', 'variant_b']:
            metrics = test['metrics'][variant]
            results['variants'][variant] = {
                'version_id': test[variant],
                'requests': metrics['requests'],
                'avg_accuracy': np.mean(metrics['accuracy']) if metrics['accuracy'] else 0,
                'avg_latency': np.mean(metrics['latency']) if metrics['latency'] else 0,
                'p95_latency': np.percentile(metrics['latency'], 95) if metrics['latency'] else 0
            }
        
        results['winner'] = self._determine_winner(results['variants'])
        
        return results
    
    def _determine_winner(self, variants: Dict) -> Optional[str]:
        if variants['variant_a']['requests'] < 100 or variants['variant_b']['requests'] < 100:
            return None
        
        from scipy import stats
        
        a_accuracy = self.active_tests[list(self.active_tests.keys())[0]]['metrics']['variant_a']['accuracy']
        b_accuracy = self.active_tests[list(self.active_tests.keys())[0]]['metrics']['variant_b']['accuracy']
        
        if len(a_accuracy) >= 30 and len(b_accuracy) >= 30:
            _, p_value = stats.ttest_ind(a_accuracy, b_accuracy)
            
            if p_value < 0.05:
                if np.mean(a_accuracy) > np.mean(b_accuracy):
                    return 'variant_a'
                else:
                    return 'variant_b'
        
        return None
    
    def complete_test(self, test_name: str, promote_winner: bool = False) -> Dict:
        results = self.get_test_results(test_name)
        
        if promote_winner and results['winner']:
            winner_version = self.active_tests[test_name][results['winner']]
            self.registry.promote_model(
                self.active_tests[test_name]['model_name'],
                self.active_tests[test_name]['language'],
                winner_version,
                'production'
            )
        
        del self.active_tests[test_name]
        
        return results


class ModelLifecycleManager:
    def __init__(self, registry: ModelRegistry, config: Dict):
        self.registry = registry
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def automated_retraining(self, model_name: str, language: str,
                           training_data: tf.data.Dataset,
                           validation_data: tf.data.Dataset,
                           min_improvement: float = 0.01) -> Optional[ModelVersion]:
        current_model, current_version = self.registry.get_model(model_name, language)
        
        from core.models import SignLanguageTransformer
        new_model = SignLanguageTransformer(
            num_classes=current_model.num_classes,
            **self.config['models']['recognition']
        )
        
        new_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = new_model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.config.get('retraining_epochs', 20)
        )
        
        new_accuracy = max(history.history['val_accuracy'])
        improvement = new_accuracy - current_version.metrics.get('accuracy', 0)
        
        if improvement >= min_improvement:
            new_version = self.registry.register_model(
                new_model,
                model_name,
                language,
                metrics={'accuracy': new_accuracy},
                parameters={'epochs': len(history.history['loss'])},
                tags=['automated_retrain']
            )
            
            self.logger.info(f"Automated retraining improved accuracy by {improvement:.2%}")
            
            return new_version
        
        self.logger.info(f"Automated retraining did not meet improvement threshold")
        return None
    
    def cleanup_old_versions(self, model_name: str, language: str,
                           keep_last_n: int = 5,
                           keep_days: int = 30):
        versions = self.registry.list_versions(model_name, language)
        
        versions_to_keep = set()
        
        for v in versions[:keep_last_n]:
            versions_to_keep.add(v.version_id)
        
        cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
        for v in versions:
            if v.created_at > cutoff_date or v.status == 'production':
                versions_to_keep.add(v.version_id)
        
        for version in versions:
            if version.version_id not in versions_to_keep:
                self._delete_version(version)
                self.logger.info(f"Deleted old version {version.version_id}")
    
    def _delete_version(self, version: ModelVersion):
        if version.path.startswith('s3://'):
            pass
        else:
            if os.path.exists(version.path):
                shutil.rmtree(version.path)
        
        metadata_path = os.path.join(
            self.registry.local_cache_dir, 'metadata',
            version.model_name, version.language,
            f"{version.version_id}.json"
        )
        if os.path.exists(metadata_path):
            os.remove(metadata_path)