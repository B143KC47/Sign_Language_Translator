import cv2
import numpy as np
import argparse
import time
from hand_detector_v2 import ImprovedHandDetector
from gesture_classifier_v2 import ImprovedGestureClassifier
from translation_engine import TranslationEngine, GestureTransitionDetector
from utils_v2 import (create_enhanced_dataset, load_enhanced_dataset, 
                     draw_enhanced_text, create_visualization_window,
                     PerformanceMonitor, augment_image_realtime)


class SignLanguageTranslator:
    def __init__(self):
        self.detector = ImprovedHandDetector(min_detection_confidence=0.8)
        self.classifier = ImprovedGestureClassifier()
        self.translation_engine = TranslationEngine()
        self.transition_detector = GestureTransitionDetector()
        self.performance_monitor = PerformanceMonitor()
        
        self.recording = False
        self.recorded_gestures = []
        
    def train_model(self, epochs=50):
        print("Creating enhanced training dataset...")
        X, y = create_enhanced_dataset()
        
        print("Training improved gesture classifier...")
        self.classifier.build_static_model(X.shape[1])
        history = self.classifier.train(X, y, epochs=epochs)
        
        self.classifier.save_models()
        print("Models saved successfully!")
        
        final_accuracy = history.history['accuracy'][-1]
        print(f"Final training accuracy: {final_accuracy:.2%}")
        
    def run_realtime_translation(self, camera_id=0, use_visualization=True):
        model_loaded = self.classifier.load_models()
        if not model_loaded:
            print("No trained model found. Training new model...")
            self.train_model()
            self.classifier.load_models()
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nControls:")
        print("- 'q': Quit")
        print("- 'c': Clear translation")
        print("- 'r': Toggle recording")
        print("- 'p': Toggle preprocessing")
        print("- 's': Save screenshot")
        
        use_preprocessing = True
        show_debug = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.performance_monitor.update()
            
            frame = cv2.flip(frame, 1)
            
            if use_preprocessing:
                frame_processed = augment_image_realtime(frame)
            else:
                frame_processed = frame
            
            start_time = time.time()
            
            results = self.detector.detect_hands(frame_processed, preprocess=use_preprocessing)
            
            landmarks = self.detector.get_landmarks(results, frame.shape)
            landmarks = self.detector.smooth_landmarks(landmarks)
            
            gesture_text = "No gesture"
            confidence = 0
            
            if landmarks:
                features = self.detector.extract_advanced_features(landmarks)
                
                if features is not None:
                    is_transition = self.transition_detector.detect_transition(features)
                    
                    if not is_transition:
                        gesture, confidence = self.classifier.predict_with_confidence_threshold(
                            features, threshold=0.6
                        )
                        
                        if gesture:
                            self.translation_engine.add_gesture(gesture, confidence)
                            self.classifier.update_sequence(features)
                            gesture_text = f"{gesture} ({confidence:.2f})"
                            
                            if self.recording:
                                self.recorded_gestures.append({
                                    'gesture': gesture,
                                    'confidence': confidence,
                                    'timestamp': time.time()
                                })
                    
                    word_gesture = self.classifier.detect_word_gesture()
                    if word_gesture:
                        gesture_text = f"WORD: {word_gesture}"
            
            processing_time = (time.time() - start_time) * 1000
            self.performance_monitor.add_processing_time(processing_time)
            
            frame = self.detector.draw_enhanced_skeleton(frame, results)
            
            completed_word = self.translation_engine.detect_word_completion()
            
            texts = [
                f"FPS: {self.performance_monitor.get_fps():.1f}",
                f"Gesture: {gesture_text}",
                f"Translation: {self.translation_engine.get_translation()}",
            ]
            
            if show_debug:
                stats = self.translation_engine.get_statistics()
                texts.extend([
                    f"Words: {stats['words_completed']}",
                    f"Avg Conf: {stats['avg_confidence']:.2f}"
                ])
            
            colors = [
                (0, 255, 255),
                (0, 255, 0) if confidence > 0.7 else (0, 165, 255),
                (255, 255, 0),
                (200, 200, 200),
                (200, 200, 200)
            ]
            
            frame = draw_enhanced_text(frame, texts, colors[:len(texts)])
            
            if self.recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            
            cv2.imshow('Advanced Sign Language Translation', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.translation_engine.clear()
                self.transition_detector.reset()
            elif key == ord('r'):
                self.recording = not self.recording
                if not self.recording and self.recorded_gestures:
                    print(f"Recording stopped. {len(self.recorded_gestures)} gestures recorded.")
            elif key == ord('p'):
                use_preprocessing = not use_preprocessing
                print(f"Preprocessing: {'ON' if use_preprocessing else 'OFF'}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.png", frame)
                print(f"Screenshot saved: screenshot_{timestamp}.png")
            elif key == ord('d'):
                show_debug = not show_debug
        
        cap.release()
        cv2.destroyAllWindows()
    
    def benchmark_performance(self):
        print("Running performance benchmark...")
        
        test_features = np.random.randn(100, 91)
        
        start_time = time.time()
        for features in test_features:
            self.classifier.predict_with_confidence_threshold(features)
        
        inference_time = (time.time() - start_time) / len(test_features) * 1000
        print(f"Average inference time: {inference_time:.2f} ms")
        print(f"Max FPS possible: {1000/inference_time:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Advanced Sign Language Translation System')
    parser.add_argument('--mode', type=str, default='demo', 
                       choices=['demo', 'train', 'benchmark'],
                       help='Mode to run the application')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    translator = SignLanguageTranslator()
    
    if args.mode == 'train':
        translator.train_model(epochs=args.epochs)
    elif args.mode == 'benchmark':
        translator.benchmark_performance()
    else:
        translator.run_realtime_translation(camera_id=args.camera)


if __name__ == "__main__":
    main()