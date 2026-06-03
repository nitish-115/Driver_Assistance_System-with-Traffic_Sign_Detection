#!/usr/bin/env python3
"""
Diagnostic Script for Traffic Sign Detector
Tests each component separately to identify issues
"""

import cv2
import numpy as np
import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_camera():
    """Test camera access"""
    print_header("TEST 1: Camera Access")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ FAILED: Cannot open camera")
            print("\nTroubleshooting:")
            print("  - Check if camera is connected")
            print("  - Try a different index: --source 1")
            print("  - Close other apps using the camera")
            return False
        
        print("✅ Camera opened successfully")
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print("❌ FAILED: Cannot read frame")
            cap.release()
            return False
        
        print(f"✅ Frame captured: {frame.shape[1]}x{frame.shape[0]}")
        
        # Test for 5 seconds
        print("\n📹 Testing camera for 5 seconds...")
        print("   Press 'Q' to quit early")
        
        frames_captured = 0
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ FAILED: Lost camera connection at frame {frames_captured}")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            frames_captured += 1
            
            # Show frame
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        fps = frames_captured / 5.0
        print(f"✅ PASSED: Captured {frames_captured} frames (~{fps:.1f} FPS)")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """Test model loading"""
    print_header("TEST 2: Model Loading")
    
    model_path = 'models/traffic_sign_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ FAILED: Model not found at {model_path}")
        print("\nTrain the model first:")
        print("  python train_model.py --epochs 10")
        return False
    
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_prediction():
    """Test model prediction on dummy data"""
    print_header("TEST 3: Model Prediction")
    
    try:
        from tensorflow import keras
        model = keras.models.load_model('models/traffic_sign_model.h5')
        
        # Create dummy image
        dummy_image = np.random.rand(1, 32, 32, 3).astype('float32')
        
        print("Testing prediction on dummy image...")
        prediction = model.predict(dummy_image, verbose=0)
        
        print(f"✅ Prediction successful")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Top class: {np.argmax(prediction)}")
        print(f"   Confidence: {np.max(prediction):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_detection():
    """Test color detection on camera"""
    print_header("TEST 4: Color Detection (ROI Finding)")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ FAILED: Cannot open camera")
            return False
        
        print("Testing color-based ROI detection for 5 seconds...")
        print("Try showing red, blue, or yellow objects to the camera")
        print("Press 'Q' to quit early")
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Color detection (same as in main detector)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Red, blue, yellow masks
            mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            mask_blue = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))
            mask_yellow = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([35, 255, 255]))
            
            mask = mask_red1 | mask_red2 | mask_blue | mask_yellow
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # Show count
            cv2.putText(frame, f"Potential regions: {len(contours)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Color Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ PASSED: Color detection working")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated():
    """Test the full detection pipeline"""
    print_header("TEST 5: Integrated Detection Pipeline")
    
    try:
        from tensorflow import keras
        import time
        
        # Load model
        model = keras.models.load_model('models/traffic_sign_model.h5')
        print("✅ Model loaded")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ FAILED: Cannot open camera")
            return False
        
        print("✅ Camera opened")
        print("\nRunning full detection for 10 seconds...")
        print("Press 'Q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ FAILED: Lost camera at frame {frame_count}")
                break
            
            frame_count += 1
            
            try:
                # Simple color detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                mask = mask_red1 | mask_red2
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process top 3 contours
                for contour in contours[:3]:
                    area = cv2.contourArea(contour)
                    if area < 500:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = frame[y:y+h, x:x+w]
                    
                    if roi.size > 0:
                        # Preprocess
                        roi_resized = cv2.resize(roi, (32, 32))
                        roi_normalized = roi_resized / 255.0
                        roi_batch = np.expand_dims(roi_normalized, axis=0)
                        
                        # Predict
                        prediction = model.predict(roi_batch, verbose=0)
                        confidence = np.max(prediction)
                        
                        if confidence > 0.85:
                            detection_count += 1
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show stats
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f} | Detections: {detection_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Integrated Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"⚠️  Warning at frame {frame_count}: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
        fps = frame_count / (time.time() - start_time)
        print(f"\n✅ PASSED: Ran for {frame_count} frames")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Total detections: {detection_count}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Traffic Sign Detector - Diagnostic Tool                 ║
    ║  Tests each component to identify issues                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "Camera Access": False,
        "Model Loading": False,
        "Model Prediction": False,
        "Color Detection": False,
        "Integrated Pipeline": False
    }
    
    # Run tests
    results["Camera Access"] = test_camera()
    results["Model Loading"] = test_model()
    
    if results["Model Loading"]:
        results["Model Prediction"] = test_model_prediction()
    
    if results["Camera Access"]:
        results["Color Detection"] = test_color_detection()
    
    if results["Camera Access"] and results["Model Loading"]:
        results["Integrated Pipeline"] = test_integrated()
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 All tests passed! Your system is ready.")
        print("\nYou can now run:")
        print("  python main_detector_optimized.py")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("  - Camera issues: Check permissions, try different camera index")
        print("  - Model issues: Train model with: python train_model.py")
        print("  - Dependencies: pip install -r requirements.txt")
    
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
