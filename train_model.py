"""
Traffic Sign Recognition Model Training
Using Transfer Learning with MobileNetV2
"""
import os
import sys

print("==== DEBUG START ====")
print("Python file:", __file__)
print("CWD:", os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "GTSRB")

print("BASE_DIR:", BASE_DIR)
print("DATASET_DIR:", DATASET_DIR)
print("DATASET EXISTS:", os.path.exists(DATASET_DIR))

if os.path.exists(DATASET_DIR):
    print("DATASET CONTENT:", os.listdir(DATASET_DIR))
else:
    print("❌ DATASET PATH INVALID")

print("==== DEBUG END ====")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm


class TrafficSignTrainer:
    def __init__(self, data_dir='data/GTSRB', img_size=(32, 32), batch_size=32):
        """
        Initialize the Traffic Sign Trainer
        
        Args:
            data_dir: Directory containing the dataset
            img_size: Input image size
            batch_size: Training batch size
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = 43  # GTSRB has 43 classes
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self):
        """Load and preprocess the GTSRB dataset"""
        print("\n📚 Loading Traffic Sign Dataset...")
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Dataset not found at {self.data_dir}")
            print("\nPlease download the GTSRB dataset:")
            print("1. Visit: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
            print("2. Download and extract to 'data/GTSRB' folder")
            return False
        
        X = []
        y = []
        
        # Load images from each class folder
        for class_id in tqdm(range(self.num_classes), desc="Loading classes"):
            class_folder = os.path.join(self.data_dir, 'Train', str(class_id))
            
            if not os.path.exists(class_folder):
                continue
            
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.png', '.jpg', '.ppm')):
                    img_path = os.path.join(class_folder, img_name)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    X.append(img)
                    y.append(class_id)
        
        X = np.array(X, dtype='float32')
        y = np.array(y)
        
        print(f"✓ Loaded {len(X)} images from {self.num_classes} classes")
        
        # Normalize pixel values
        X = X / 255.0
        
        # Split data: 70% train, 15% validation, 15% test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
        )
        
        print(f"✓ Training set: {len(self.X_train)} images")
        print(f"✓ Validation set: {len(self.X_val)} images")
        print(f"✓ Test set: {len(self.X_test)} images")
        
        return True
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            shear_range=0.15,
            brightness_range=[0.8, 1.2],
            horizontal_flip=False,  # Traffic signs shouldn't be flipped
            fill_mode='nearest'
        )
        return datagen
    
    def build_model(self, use_transfer_learning=True):
        """
        Build the traffic sign classification model
        
        Args:
            use_transfer_learning: Whether to use transfer learning with MobileNetV2
        """
        print("\n🏗️  Building Model...")
        
        if use_transfer_learning:
            # Load pre-trained MobileNetV2 (without top layers)
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Build model on top
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            print("✓ Using Transfer Learning with MobileNetV2")
        else:
            # Custom CNN model
            model = keras.Sequential([
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fully Connected Layers
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            print("✓ Using Custom CNN Architecture")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n📊 Model Summary:")
        model.summary()
        
        return model
    
    def train_model(self, model, epochs=30, save_path='models/traffic_sign_model.h5'):
        """Train the model with callbacks"""
        print("\n🎯 Training Model...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Data augmentation
        datagen = self.create_data_augmentation()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed!")
        return history
    
    def evaluate_model(self, model):
        """Evaluate model on test set"""
        print("\n📈 Evaluating Model...")
        
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"✓ Test Loss: {test_loss:.4f}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Make predictions
        predictions = model.predict(self.X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate per-class accuracy
        from sklearn.metrics import classification_report
        
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, predicted_classes, 
                                   target_names=[f"Class {i}" for i in range(self.num_classes)]))
        
        return test_accuracy
    
    def plot_training_history(self, history, save_path='models/training_history.png'):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history saved to {save_path}")
        plt.close()
    
    def save_training_data(self, save_path='models/training_data.pkl'):
        """Save preprocessed training data for future use"""
        data = {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Training data saved to {save_path}")


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Traffic Sign Recognition Model')
    parser.add_argument('--data_dir', type=str, default='data/GTSRB',
                       help='Path to GTSRB dataset')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--transfer_learning', action='store_true', default=True,
                       help='Use transfer learning with MobileNetV2')
    parser.add_argument('--model_path', type=str, default='models/traffic_sign_model.h5',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TrafficSignTrainer(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Load data
    if not trainer.load_data():
        return
    
    # Build model
    model = trainer.build_model(use_transfer_learning=args.transfer_learning)
    
    # Train model
    history = trainer.train_model(model, epochs=args.epochs, save_path=args.model_path)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate model
    trainer.evaluate_model(model)
    
    # Save training data
    trainer.save_training_data()
    
    print("\n" + "="*50)
    print("🎉 Training Pipeline Completed Successfully!")
    print("="*50)
    print(f"\nModel saved to: {args.model_path}")
    print("You can now run the detector using:")
    print(f"  python main_detector.py --model {args.model_path}")


if __name__ == "__main__":
    main()
