import cv2
import numpy as np
import argparse
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
from utils import TranslationBuffer, draw_text, create_sample_dataset, load_dataset


def main(mode='demo', camera_id=0):
    detector = HandDetector(min_detection_confidence=0.7)
    classifier = GestureClassifier()
    translation_buffer = TranslationBuffer()
    
    if mode == 'train':
        print("Creating sample dataset...")
        X, y = create_sample_dataset()
        
        print("Training gesture classifier...")
        classifier.build_model(63)
        history = classifier.train(X, y, epochs=30)
        
        classifier.save_model('gesture_model.h5')
        print("Model saved as 'gesture_model.h5'")
        return
    
    model_loaded = classifier.load_model('gesture_model.h5')
    if not model_loaded:
        print("No trained model found. Run with --mode train first.")
        print("Creating and training a sample model...")
        X, y = create_sample_dataset()
        classifier.build_model(63)
        classifier.train(X, y, epochs=30)
        classifier.save_model('gesture_model.h5')
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 'c' to clear translation")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        results = detector.detect_hands(frame)
        
        landmarks = detector.get_landmarks(results, frame.shape)
        
        if landmarks:
            features = detector.extract_hand_features(landmarks)
            if features is not None:
                gesture, confidence = classifier.predict(features)
                translation_buffer.add_gesture(gesture, confidence)
                
                draw_text(frame, f"Gesture: {gesture} ({confidence:.2f})", 
                         (10, 30), font_scale=0.7, color=(0, 255, 0))
        
        frame = detector.draw_skeleton(frame, results)
        
        translation = translation_buffer.get_translation()
        if translation:
            draw_text(frame, f"Translation: {translation}", 
                     (10, 60), font_scale=0.7, color=(255, 255, 0))
        
        draw_text(frame, "Press 'q' to quit, 'c' to clear", 
                 (10, frame.shape[0] - 20), font_scale=0.5, color=(200, 200, 200))
        
        cv2.imshow('Hand Sign Language Translation', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            translation_buffer.clear()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Sign Language Detection and Translation')
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'train'],
                       help='Mode to run the application in')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    
    args = parser.parse_args()
    main(mode=args.mode, camera_id=args.camera)