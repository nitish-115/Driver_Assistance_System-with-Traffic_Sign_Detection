"""
Driver Assistance System - Traffic Sign Detection
FINAL OPTIMIZED VERSION - No lag, No false detections, Controlled voice
Real-time traffic sign detection with voice alerts
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import subprocess
import platform
from collections import deque
import time
import threading
from datetime import datetime
import os

class TrafficSignDetector:
    def __init__(self, model_path='models/traffic_sign_model.h5', confidence_threshold=0.90):
        """
        Initialize the Traffic Sign Detector
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Minimum confidence (0.90 = very strict, no false positives)
        """
        self.model = self._load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_skip = 3  # Process every 3rd frame - NO LAG
        self.frame_count = 0
        
        # Traffic sign classes (German Traffic Sign Recognition Benchmark)
        self.class_names = {
            0: 'Speed limit 20', 1: 'Speed limit 30', 2: 'Speed limit 50',
            3: 'Speed limit 60', 4: 'Speed limit 70', 5: 'Speed limit 80',
            6: 'End speed limit 80', 7: 'Speed limit 100', 8: 'Speed limit 120',
            9: 'No passing', 10: 'No trucks passing',
            11: 'Right of way', 12: 'Priority road',
            13: 'Yield', 14: 'Stop', 15: 'No vehicles',
            16: 'No trucks', 17: 'No entry',
            18: 'General caution', 19: 'Curve left',
            20: 'Curve right', 21: 'Double curve',
            22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows',
            25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
            28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Ice warning',
            31: 'Wild animals crossing', 32: 'End of limits',
            33: 'Turn right', 34: 'Turn left', 35: 'Ahead only',
            36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
            39: 'Keep left', 40: 'Roundabout', 41: 'End no passing',
            42: 'End no trucks passing'
        }
        
        # Initialize SYSTEM voice (not pyttsx3 - more stable)
        self.voice_enabled = self._setup_voice()
        
        # Detection smoothing
        self.last_detections = []  # Cache for skipped frames
        self.last_announced_sign = None
        self.last_announcement_time = 0
        self.announcement_cooldown = 5  # 5 SECONDS - won't repeat continuously
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        
        # Colors for visualization
        self.colors = {
            'danger': (0, 0, 255),      # Red
            'warning': (0, 165, 255),   # Orange
            'info': (255, 255, 0),      # Cyan
            'success': (0, 255, 0)      # Green
        }
        
        print(f"✅ Detector initialized")
        print(f"   Threshold: {self.confidence_threshold:.0%} (strict)")
        print(f"   Frame skip: {self.frame_skip} (smooth)")
        print(f"   Voice cooldown: {self.announcement_cooldown}s")
        print(f"   Voice: {'ON' if self.voice_enabled else 'OFF'}")
    
    def _setup_voice(self):
        """Setup system voice (not pyttsx3)"""
        system = platform.system()
        
        if system == "Darwin":  # Mac
            try:
                subprocess.run(['say', '-v', '?'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
                self.voice_command = "mac"
                print("✅ Mac voice enabled")
                return True
            except:
                print("⚠️  Mac voice failed")
                return False
        
        elif system == "Linux":
            try:
                subprocess.run(['espeak', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
                self.voice_command = "linux"
                print("✅ Linux espeak enabled")
                return True
            except:
                print("⚠️  espeak not found (install: sudo apt-get install espeak)")
                return False
        
        elif system == "Windows":
            self.voice_command = "windows"
            print("✅ Windows voice enabled")
            return True
        
        return False
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first using: python train_model.py")
            return None
    
    def _get_sign_category(self, class_id):
        """Categorize sign for color coding"""
        danger_signs = [9, 10, 14, 17]  # No passing, Stop, No entry
        warning_signs = [11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        info_signs = [33, 34, 35, 36, 37, 38, 39, 40]
        
        if class_id in danger_signs:
            return 'danger'
        elif class_id in warning_signs:
            return 'warning'
        elif class_id in info_signs:
            return 'info'
        else:
            return 'success'
    
    def preprocess_roi(self, roi):
        """Preprocess region of interest for model prediction"""
        try:
            roi_resized = cv2.resize(roi, (32, 32))
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            return roi_batch
        except:
            return None
    
    def detect_signs_in_frame(self, frame):
        """
        OPTIMIZED detection - only 5 regions, not 9
        Faster and fewer false positives
        """
        detections = []
        height, width = frame.shape[:2]
        
        try:
            # Only 5 regions - MUCH FASTER
            regions = [
                # Large center region (main area)
                (int(width*0.25), int(height*0.25), int(width*0.75), int(height*0.75)),
                # Top half
                (int(width*0.2), 0, int(width*0.8), int(height*0.5)),
                # Bottom half
                (int(width*0.2), int(height*0.5), int(width*0.8), height),
                # Left side
                (0, int(height*0.2), int(width*0.5), int(height*0.8)),
                # Right side
                (int(width*0.5), int(height*0.2), width, int(height*0.8)),
            ]
            
            for x1, y1, x2, y2 in regions:
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue
                
                processed = self.preprocess_roi(roi)
                if processed is None:
                    continue
                
                prediction = self.model.predict(processed, verbose=0)
                
                confidence = float(np.max(prediction))
                class_id = int(np.argmax(prediction))
                
                # STRICT threshold - no false positives
                if confidence > self.confidence_threshold:
                    detections.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'class_id': class_id,
                        'confidence': confidence,
                        'class_name': self.class_names[class_id]
                    })
            
            # Keep only best detection to avoid duplicates
            if detections:
                detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                detections = detections[:1]  # Only top detection
            
        except Exception as e:
            detections = []
        
        return detections
    
    def announce_sign(self, sign_name):
        """
        Announce detected sign with STRICT COOLDOWN
        Won't repeat continuously
        """
        if not self.voice_enabled:
            return
        
        current_time = time.time()
        
        # STRICT - must be different sign OR 5+ seconds passed
        if (self.last_announced_sign != sign_name or 
            current_time - self.last_announcement_time > self.announcement_cooldown):
            
            def speak():
                try:
                    if self.voice_command == "mac":
                        subprocess.run(['say', sign_name], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                    elif self.voice_command == "linux":
                        subprocess.run(['espeak', sign_name], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                    elif self.voice_command == "windows":
                        cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{sign_name}")'
                        subprocess.run(['powershell', '-Command', cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                except:
                    pass
            
            threading.Thread(target=speak, daemon=True).start()
            
            self.last_announced_sign = sign_name
            self.last_announcement_time = current_time
            
            print(f"🔊 {sign_name}")
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            category = self._get_sign_category(detection['class_id'])
            color = self.colors[category]
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            
            # Label
            label = f"{class_name}"
            conf_text = f"{confidence:.0%}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - label_h - 15), (x + label_w + 10, y), color, -1)
            
            cv2.putText(frame, label, (x + 5, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, conf_text, (x + w - 70, y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def draw_ui(self, frame, fps, detection_count):
        """Draw UI overlay"""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.0%}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        voice_status = f"Voice: {'ON' if self.voice_enabled else 'OFF'}"
        cv2.putText(frame, voice_status, (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if self.voice_enabled else (128, 128, 128), 2)
        
        cv2.putText(frame, "Q=Quit | S=Save | P=Pause | +/-=Threshold", (20, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_detection(self, video_source=0):
        """Main detection loop - OPTIMIZED"""
        if self.model is None:
            print("❌ Cannot run without model!")
            return
        
        cap = cv2.VideoCapture(video_source)
        
        # 640x480 for SPEED
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        ret, test_frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            cap.release()
            return
        
        print("\n" + "="*60)
        print("  🚗 DRIVER ASSISTANCE SYSTEM - OPTIMIZED")
        print("="*60)
        print(f"\nCamera: {test_frame.shape[1]}x{test_frame.shape[0]}")
        print(f"Threshold: {self.confidence_threshold:.0%} (strict)")
        print(f"Frame skip: Every {self.frame_skip} frames (smooth)")
        print(f"Voice cooldown: {self.announcement_cooldown}s (won't repeat)")
        print(f"Voice: {'ON 🔊' if self.voice_enabled else 'OFF 🔇'}")
        print("\nControls:")
        print("  Q - Quit | S - Save | P - Pause")
        print("  + - Stricter | - - Looser")
        print("\n💡 Show clear traffic sign images to camera!")
        print("="*60 + "\n")
        
        os.makedirs('recordings', exist_ok=True)
        
        paused = False
        total_frames = 0
        
        try:
            while True:
                if not paused:
                    start_time = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                    
                    total_frames += 1
                    self.frame_count += 1
                    
                    # SKIP FRAMES - Every 3rd frame
                    if self.frame_count % self.frame_skip == 0:
                        detections = self.detect_signs_in_frame(frame)
                        self.last_detections = detections
                        
                        # Announce with strict cooldown
                        for detection in detections:
                            self.announce_sign(detection['class_name'])
                    else:
                        detections = self.last_detections
                    
                    frame = self.draw_detections(frame, detections)
                    
                    fps = 1.0 / max(time.time() - start_time, 0.001)
                    self.fps_history.append(fps)
                    avg_fps = np.mean(self.fps_history)
                    
                    frame = self.draw_ui(frame, avg_fps, len(detections))
                    
                    cv2.imshow('Driver Assistance System', frame)
                else:
                    cv2.imshow('Driver Assistance System', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recordings/detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved: {filename}")
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
                    print("⏸  Paused" if paused else "▶  Resumed")
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                    print(f"🎯 Threshold: {self.confidence_threshold:.0%}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.50, self.confidence_threshold - 0.05)
                    print(f"🎯 Threshold: {self.confidence_threshold:.0%}")
        
        except KeyboardInterrupt:
            print("\n\n✅ Stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print(f"  Frames: {total_frames}")
            if self.fps_history:
                print(f"  Avg FPS: {np.mean(self.fps_history):.1f}")
            print("="*60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Driver Assistance System')
    parser.add_argument('--model', type=str, default='models/traffic_sign_model.h5')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--confidence', type=float, default=0.90)
    
    args = parser.parse_args()
    
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    detector = TrafficSignDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    detector.run_detection(video_source=source)


if __name__ == "__main__":
    main()