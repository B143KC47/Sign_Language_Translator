import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Info
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import boto3
from functools import wraps
import asyncio
import psutil
import numpy as np


class ProductionMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self._setup_metrics()
        self._setup_tracing()
        
        self.cloudwatch = boto3.client('cloudwatch') if config.get('use_cloudwatch') else None
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('sign_language_translator')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"service": "%(name)s", "message": "%(message)s", '
            '"trace_id": "%(trace_id)s", "span_id": "%(span_id)s"}'
        )
        
        handlers = []
        
        if 'stdout' in self.config['logging']['outputs']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)
        
        if 'file' in self.config['logging']['outputs']:
            file_handler = logging.FileHandler('app.log')
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        for handler in handlers:
            logger.addHandler(handler)
        
        LoggingInstrumentor().instrument()
        
        return logger
    
    def _setup_metrics(self):
        self.metrics = {
            'requests_total': Counter(
                'sign_language_requests_total',
                'Total number of translation requests',
                ['language', 'endpoint', 'status']
            ),
            'request_duration': Histogram(
                'sign_language_request_duration_seconds',
                'Request duration in seconds',
                ['language', 'endpoint'],
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
            ),
            'model_inference_time': Histogram(
                'model_inference_duration_seconds',
                'Model inference duration in seconds',
                ['model_name', 'language'],
                buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
            ),
            'active_connections': Gauge(
                'sign_language_active_connections',
                'Number of active connections'
            ),
            'translation_accuracy': Gauge(
                'sign_language_translation_accuracy',
                'Translation accuracy score',
                ['language']
            ),
            'cache_hits': Counter(
                'sign_language_cache_hits_total',
                'Total number of cache hits',
                ['cache_type']
            ),
            'cache_misses': Counter(
                'sign_language_cache_misses_total',
                'Total number of cache misses',
                ['cache_type']
            ),
            'error_rate': Counter(
                'sign_language_errors_total',
                'Total number of errors',
                ['error_type', 'language']
            ),
            'model_info': Info(
                'sign_language_model',
                'Information about loaded models'
            )
        }
        
        self.metrics['cpu_usage'] = Gauge(
            'sign_language_cpu_usage_percent',
            'CPU usage percentage'
        )
        self.metrics['memory_usage'] = Gauge(
            'sign_language_memory_usage_bytes',
            'Memory usage in bytes'
        )
        self.metrics['gpu_usage'] = Gauge(
            'sign_language_gpu_usage_percent',
            'GPU usage percentage'
        )
    
    def _setup_tracing(self):
        if not self.config['tracing']['enabled']:
            return
        
        trace.set_tracer_provider(TracerProvider())
        
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config['tracing'].get('jaeger_host', 'localhost'),
            agent_port=self.config['tracing'].get('jaeger_port', 6831),
            service_name='sign_language_translator'
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
    
    def track_request(self, endpoint: str, language: str):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                status = 'success'
                
                with self.tracer.start_as_current_span(f"{endpoint}_request") as span:
                    span.set_attribute("language", language)
                    span.set_attribute("endpoint", endpoint)
                    
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        status = 'error'
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                        self.metrics['error_rate'].labels(
                            error_type=type(e).__name__,
                            language=language
                        ).inc()
                        raise
                    finally:
                        duration = time.time() - start_time
                        self.metrics['requests_total'].labels(
                            language=language,
                            endpoint=endpoint,
                            status=status
                        ).inc()
                        self.metrics['request_duration'].labels(
                            language=language,
                            endpoint=endpoint
                        ).observe(duration)
                        
                        self._send_cloudwatch_metric(
                            'RequestDuration',
                            duration,
                            'Seconds',
                            {'Language': language, 'Endpoint': endpoint}
                        )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                status = 'success'
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = 'error'
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics['requests_total'].labels(
                        language=language,
                        endpoint=endpoint,
                        status=status
                    ).inc()
                    self.metrics['request_duration'].labels(
                        language=language,
                        endpoint=endpoint
                    ).observe(duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def track_model_inference(self, model_name: str, language: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    inference_time = time.time() - start_time
                    self.metrics['model_inference_time'].labels(
                        model_name=model_name,
                        language=language
                    ).observe(inference_time)
                    
                    return result
                except Exception as e:
                    self.logger.error(f"Model inference failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def track_cache(self, cache_type: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                if result is not None:
                    self.metrics['cache_hits'].labels(cache_type=cache_type).inc()
                else:
                    self.metrics['cache_misses'].labels(cache_type=cache_type).inc()
                
                return result
            return wrapper
        return decorator
    
    def update_system_metrics(self):
        self.metrics['cpu_usage'].set(psutil.cpu_percent())
        self.metrics['memory_usage'].set(psutil.virtual_memory().used)
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = np.mean([gpu.load * 100 for gpu in gpus])
                self.metrics['gpu_usage'].set(gpu_usage)
        except:
            pass
    
    def log_translation_quality(self, language: str, accuracy: float):
        self.metrics['translation_accuracy'].labels(language=language).set(accuracy)
        
        self.logger.info(
            f"Translation quality update",
            extra={
                'language': language,
                'accuracy': accuracy,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _send_cloudwatch_metric(self, metric_name: str, value: float, 
                               unit: str, dimensions: Dict[str, str]):
        if not self.cloudwatch:
            return
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace='SignLanguageTranslator',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Dimensions': [
                            {'Name': k, 'Value': v} 
                            for k, v in dimensions.items()
                        ],
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to send CloudWatch metric: {e}")
    
    def create_dashboard(self):
        dashboard_config = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["SignLanguageTranslator", "RequestDuration", 
                             {"stat": "Average"}],
                            [".", ".", {"stat": "p99"}]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "Request Duration"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["SignLanguageTranslator", "TranslationAccuracy",
                             {"stat": "Average"}]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "Translation Accuracy"
                    }
                }
            ]
        }
        
        return dashboard_config


class PerformanceProfiler:
    def __init__(self):
        self.profiles = {}
        
    def profile_function(self, name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import cProfile
                import pstats
                import io
                
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profiler.disable()
                    
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats(20)
                    
                    self.profiles[name] = {
                        'timestamp': datetime.utcnow(),
                        'stats': s.getvalue()
                    }
            
            return wrapper
        return decorator
    
    def get_profile_report(self, name: str) -> Optional[str]:
        return self.profiles.get(name, {}).get('stats')


class AlertManager:
    def __init__(self, config: Dict):
        self.config = config
        self.alert_thresholds = {
            'error_rate': 0.05,
            'response_time_p99': 2.0,
            'cpu_usage': 80,
            'memory_usage': 90,
            'accuracy_drop': 0.1
        }
        self.alert_history = {}
        
    def check_alerts(self, metrics: Dict):
        alerts = []
        
        if metrics.get('error_rate', 0) > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'HIGH_ERROR_RATE',
                'severity': 'critical',
                'message': f"Error rate {metrics['error_rate']:.2%} exceeds threshold",
                'value': metrics['error_rate']
            })
        
        if metrics.get('response_time_p99', 0) > self.alert_thresholds['response_time_p99']:
            alerts.append({
                'type': 'SLOW_RESPONSE',
                'severity': 'warning',
                'message': f"P99 response time {metrics['response_time_p99']:.2f}s exceeds threshold",
                'value': metrics['response_time_p99']
            })
        
        for alert in alerts:
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: Dict):
        alert_key = f"{alert['type']}_{alert['severity']}"
        
        last_alert_time = self.alert_history.get(alert_key)
        if last_alert_time and (datetime.utcnow() - last_alert_time).seconds < 300:
            return
        
        self.alert_history[alert_key] = datetime.utcnow()
        
        logging.getLogger(__name__).error(f"ALERT: {alert['message']}")
        
        if self.config.get('sns_topic_arn'):
            sns = boto3.client('sns')
            sns.publish(
                TopicArn=self.config['sns_topic_arn'],
                Subject=f"Sign Language Translator Alert: {alert['type']}",
                Message=json.dumps(alert, indent=2)
            )