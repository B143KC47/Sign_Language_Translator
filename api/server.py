from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
import jwt
import asyncio
from datetime import datetime, timedelta
import json
import io
from contextlib import asynccontextmanager
import logging
import yaml

from core.models import MultiLanguageSignModel
from core.translation_service import MultiLanguageTranslator, TextToSpeech
from core.data_pipeline import DataCollector, ActiveLearning
from core.monitoring import ProductionMonitor
from model.detectors.hand_detector_v2 import ImprovedHandDetector


security = HTTPBearer()
app = FastAPI(title="Sign Language Translation API", version="1.0.0")

with open('config/production_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

monitor = ProductionMonitor(config['monitoring'])
model_manager = MultiLanguageSignModel(config)
translator = MultiLanguageTranslator(config['translation'])
tts_service = TextToSpeech(config['translation'])
data_collector = DataCollector(storage_backend=config['data']['collection'].get('backend', 's3'))
active_learner = ActiveLearning()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        monitor.metrics['active_connections'].set(len(self.active_connections))
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            monitor.metrics['active_connections'].set(len(self.active_connections))
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


connection_manager = ConnectionManager()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, 
            config['api']['authentication']['jwt_secret'], 
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Sign Language Translation API")
    
    for lang_config in config['languages']['sign_languages']:
        try:
            model_manager.load_model(lang_config['code'])
            monitor.metrics['model_info'].info({
                'language': lang_config['code'],
                'version': '1.0',
                'vocabulary_size': str(lang_config['vocabulary_size'])
            })
        except Exception as e:
            logger.error(f"Failed to load model for {lang_config['code']}: {e}")
    
    asyncio.create_task(system_metrics_updater())


async def system_metrics_updater():
    while True:
        monitor.update_system_metrics()
        await asyncio.sleep(60)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(model_manager.models.keys())
    }


@app.post("/translate/image")
@monitor.track_request("translate_image", "multi")
async def translate_from_image(
    file: UploadFile = File(...),
    source_language: str = "ASL",
    target_language: str = "en",
    user_data: Dict = Depends(verify_token)
):
    if source_language not in [lang['code'] for lang in config['languages']['sign_languages']]:
        raise HTTPException(status_code=400, detail="Unsupported source language")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    detector = ImprovedHandDetector()
    results = detector.detect_hands(image)
    landmarks = detector.get_landmarks(results, image.shape)
    
    if not landmarks:
        raise HTTPException(status_code=400, detail="No hands detected in image")
    
    features = detector.extract_advanced_features(landmarks)
    
    prediction = model_manager.predict(features[np.newaxis, :], source_language)
    
    if active_learner.should_collect_sample(
        prediction['probabilities'], 
        features
    ):
        data_collector.collect_sample(
            features,
            None,
            {
                'user_id': user_data.get('user_id'),
                'source_language': source_language,
                'confidence': prediction['confidence']
            }
        )
    
    translation_result = await translator.translate_sign_sequence(
        [prediction['predicted_class']],
        source_language,
        target_language
    )
    
    return {
        "prediction": prediction,
        "translation": translation_result,
        "processing_time_ms": 0
    }


@app.post("/translate/video")
@monitor.track_request("translate_video", "multi")
async def translate_from_video(
    file: UploadFile = File(...),
    source_language: str = "ASL",
    target_language: str = "en",
    user_data: Dict = Depends(verify_token)
):
    contents = await file.read()
    
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    
    cap = cv2.VideoCapture(temp_path)
    detector = ImprovedHandDetector()
    
    all_predictions = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 5 == 0:
            results = detector.detect_hands(frame)
            landmarks = detector.get_landmarks(results, frame.shape)
            
            if landmarks:
                features = detector.extract_advanced_features(landmarks)
                prediction = model_manager.predict(features[np.newaxis, :], source_language)
                all_predictions.append(prediction['predicted_class'])
        
        frame_count += 1
    
    cap.release()
    
    translation_result = await translator.translate_sign_sequence(
        all_predictions,
        source_language,
        target_language
    )
    
    return {
        "frame_count": frame_count,
        "predictions_count": len(all_predictions),
        "translation": translation_result
    }


@app.websocket("/ws/translate/{client_id}")
async def websocket_translate(
    websocket: WebSocket, 
    client_id: str,
    source_language: str = "ASL",
    target_language: str = "en"
):
    await connection_manager.connect(websocket, client_id)
    
    detector = ImprovedHandDetector()
    prediction_queue = asyncio.Queue()
    translation_queue = await translator.translate_continuous_stream(
        prediction_queue,
        source_language,
        target_language
    )
    
    async def send_translations():
        while True:
            try:
                translation = await translation_queue.get()
                await websocket.send_json({
                    "type": "translation",
                    "data": translation
                })
            except Exception as e:
                logger.error(f"Translation send error: {e}")
                break
    
    translation_task = asyncio.create_task(send_translations())
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            results = detector.detect_hands(image)
            landmarks = detector.get_landmarks(results, image.shape)
            
            if landmarks:
                features = detector.extract_advanced_features(landmarks)
                prediction = model_manager.predict(features[np.newaxis, :], source_language)
                
                await prediction_queue.put(prediction['predicted_class'])
                
                await websocket.send_json({
                    "type": "prediction",
                    "data": {
                        "class": prediction['predicted_class'],
                        "confidence": float(prediction['confidence'])
                    }
                })
    
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        await prediction_queue.put(None)
        translation_task.cancel()
        logger.info(f"Client {client_id} disconnected")


@app.post("/tts")
async def text_to_speech(
    text: str,
    language: str = "en-US",
    voice: Optional[str] = None,
    user_data: Dict = Depends(verify_token)
):
    try:
        audio_data = await tts_service.synthesize_speech(text, language, voice)
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages")
async def get_supported_languages():
    return {
        "sign_languages": [
            {
                "code": lang['code'],
                "name": lang['name'],
                "regions": lang['regions']
            }
            for lang in config['languages']['sign_languages']
        ],
        "spoken_languages": config['languages']['spoken_languages']
    }


@app.post("/feedback")
async def submit_feedback(
    prediction_id: str,
    correct_label: Optional[int] = None,
    feedback_text: Optional[str] = None,
    user_data: Dict = Depends(verify_token)
):
    feedback_data = {
        "prediction_id": prediction_id,
        "correct_label": correct_label,
        "feedback_text": feedback_text,
        "user_id": user_data.get("user_id"),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    data_collector.collect_sample(
        np.array([]),
        correct_label,
        feedback_data
    )
    
    return {"status": "feedback_received"}


@app.get("/metrics")
async def get_metrics(user_data: Dict = Depends(verify_token)):
    if user_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/models/reload")
async def reload_models(
    language: Optional[str] = None,
    user_data: Dict = Depends(verify_token)
):
    if user_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if language:
        model_manager.models.pop(language, None)
        model_manager.load_model(language)
        return {"status": f"Model {language} reloaded"}
    else:
        for lang in model_manager.models.keys():
            model_manager.models.pop(lang, None)
            model_manager.load_model(lang)
        return {"status": "All models reloaded"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['app']['debug'],
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )