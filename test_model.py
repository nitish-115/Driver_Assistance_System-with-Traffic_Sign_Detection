"""
Model Testing and Evaluation Script
Test trained model on individual images and webcam
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from datetime import datetime


class ModelTester:
    def __init__(self, model_path='models/traffic_sign_model.h5'):
        """Initialize model tester"""
        self.model_path = model_path
        self.model = None
        
        # Class names
        self.class_names = {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
            3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
            6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
            9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
            11: 'Right-of-way at the next intersection', 12: 'Priority road',
            13: 'Yield', 14: 'Stop', 15: 'No vehicles',
            16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry',
            18: 'General caution', 19: 'Dangerous curve to the left',
            20: 'Dangerous curve to the right', 21: 'Double curve',
            22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
            25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
            28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
            31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
            33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
            36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
            39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
            42: 'End of no passing by vehicles over 3.5 metric tons'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first using train_model.py")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Resize to 32x32
        image_resized = cv2.resize(image, (32, 32))
        # Normalize
        image_normalized = image_resized / 255.0
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        return image_batch
    
    def predict_sign(self, image):
        """Predict traffic sign in image"""
        if self.model is None:
            return None, None, None
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        
        # Get top prediction
        confidence = np.max(predictions)
        class_id = np.argmax(predictions)
        class_name = self.class_names[class_id]
        
        return class_id, class_name, confidence
    
    def test_single_image(self, image_path, save_result=True):
        """Test model on a single image"""
        print(f"\n🔍 Testing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        class_id, class_name, confidence = self.predict_sign(image_rgb)
        
        if class_id is not None:
            print(f"✓ Prediction: {class_name}")
            print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"  Class ID: {class_id}")
            
            # Visualize
            if save_result:
                self.visualize_prediction(image_rgb, class_name, confidence)
        
        return class_id, class_name, confidence
    
    def visualize_prediction(self, image, class_name, confidence, save_path=None):
        """Visualize prediction result"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Prediction
        prediction_text = f"Predicted: {class_name}\nConfidence: {confidence:.2%}"
        ax2.text(0.5, 0.5, prediction_text, 
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"recordings/prediction_{timestamp}.png"
        
        os.makedirs('recordings', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Result saved to {save_path}")
        plt.close()
    
    def test_webcam_quick(self, duration=10):
        """Quick webcam test (limited duration)"""
        print(f"\n📹 Starting webcam test ({duration} seconds)")
        print("Press 'Q' to quit early")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check duration
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > duration:
                break
            
            # Predict
            class_id, class_name, confidence = self.predict_sign(frame)
            
            if confidence > 0.7:  # Show only confident predictions
                # Draw result
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw timer
            remaining = int(duration - elapsed)
            cv2.putText(frame, f"Time: {remaining}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Webcam Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam test completed")
    
    def get_top_k_predictions(self, image, k=5):
        """Get top-k predictions for an image"""
        if self.model is None:
            return []
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top-k
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'class_id': idx,
                'class_name': self.class_names[idx],
                'confidence': predictions[idx]
            })
        
        return results
    
    def batch_test_images(self, image_dir, confidence_threshold=0.7):
        """Test model on a batch of images"""
        print(f"\n📦 Batch testing images in: {image_dir}")
        
        if not os.path.exists(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            return
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg', '.ppm'))]
        
        if not image_files:
            print("❌ No images found")
            return
        
        print(f"Found {len(image_files)} images")
        
        results = []
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                class_id, class_name, confidence = self.predict_sign(image_rgb)
                
                results.append({
                    'file': img_file,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
                
                print(f"  {img_file}: {class_name} ({confidence:.2%})")
        
        # Summary
        print(f"\n📊 Batch Test Summary:")
        print(f"  Total images: {len(results)}")
        confident = sum(1 for r in results if r['confidence'] > confidence_threshold)
        print(f"  High confidence (>{confidence_threshold}): {confident}")
        print(f"  Average confidence: {np.mean([r['confidence'] for r in results]):.2%}")
        
        return results


def main():
    """Main testing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Traffic Sign Recognition Model')
    parser.add_argument('--model', type=str, default='models/traffic_sign_model.h5',
                       help='Path to trained model')
    parser.add_argument('--action', type=str, default='image',
                       choices=['image', 'webcam', 'batch'],
                       help='Test action')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image')
    parser.add_argument('--image_dir', type=str, default='data/test_images',
                       help='Directory containing test images')
    parser.add_argument('--duration', type=int, default=10,
                       help='Webcam test duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(model_path=args.model)
    
    if tester.model is None:
        return
    
    # Perform action
    if args.action == 'image':
        if args.image is None:
            print("❌ Please specify --image path")
            return
        tester.test_single_image(args.image)
    
    elif args.action == 'webcam':
        tester.test_webcam_quick(duration=args.duration)
    
    elif args.action == 'batch':
        tester.batch_test_images(args.image_dir)
    
    print("\n✓ Testing completed!")


if __name__ == "__main__":
    main()
