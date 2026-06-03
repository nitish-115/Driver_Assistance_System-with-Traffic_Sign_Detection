#!/usr/bin/env python3
"""
Quick Start Guide for Driver Assistance System
Interactive setup and testing script
"""

import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = ['tensorflow', 'cv2', 'numpy', 'pyttsx3', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall them using:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_dataset():
    """Check if dataset exists"""
    print_header("Checking Dataset")
    
    dataset_path = 'data/GTSRB/Train'
    
    if os.path.exists(dataset_path):
        # Count classes
        try:
            classes = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
            print(f"✓ Dataset found with {len(classes)} classes")
            return True
        except:
            pass
    
    print("✗ Dataset not found")
    print("\n📥 Download Instructions:")
    print("  1. Visit: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    print("  2. Download and extract to 'data/GTSRB' directory")
    print("\nOr run:")
    print("  python utils/dataset_utils.py")
    return False


def check_model():
    """Check if trained model exists"""
    print_header("Checking Trained Model")
    
    model_path = 'models/traffic_sign_model.h5'
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model found ({size_mb:.2f} MB)")
        return True
    
    print("✗ Model not found")
    print("\nTrain the model using:")
    print("  python train_model.py --epochs 30")
    return False


def setup_directories():
    """Create necessary directories"""
    print_header("Setting Up Directories")
    
    dirs = ['models', 'data', 'recordings', 'utils']
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}/")
    
    print("\n✓ Directory structure created!")


def show_menu():
    """Show interactive menu"""
    print_header("Driver Assistance System - Quick Start")
    
    print("\nWhat would you like to do?\n")
    print("1. Check system setup")
    print("2. Download dataset (instructions)")
    print("3. Train model")
    print("4. Test model on image")
    print("5. Run real-time detection")
    print("6. View dataset statistics")
    print("7. Exit")
    
    return input("\nEnter choice (1-7): ").strip()


def run_command(cmd, description):
    """Run a system command"""
    print(f"\n🚀 {description}...")
    print(f"Command: {cmd}\n")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ Error running command")
        return False


def main():
    """Main quick start function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🚗 Driver Assistance System - Traffic Sign Detection   ║
    ║        Quick Start Guide & System Setup                  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # System setup check
            setup_directories()
            deps_ok = check_dependencies()
            data_ok = check_dataset()
            model_ok = check_model()
            
            print_header("Setup Summary")
            print(f"Dependencies: {'✓ Ready' if deps_ok else '✗ Missing'}")
            print(f"Dataset:      {'✓ Ready' if data_ok else '✗ Missing'}")
            print(f"Model:        {'✓ Ready' if model_ok else '✗ Not trained'}")
            
            if deps_ok and data_ok and model_ok:
                print("\n🎉 System is ready! You can run real-time detection.")
            else:
                print("\n⚠️  Please complete the missing steps above.")
        
        elif choice == '2':
            # Dataset download
            print_header("Dataset Download Instructions")
            print("\n📥 GTSRB Dataset (German Traffic Sign Recognition Benchmark)")
            print("\nStep 1: Download")
            print("  Visit: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
            print("  Click 'Download' and save the zip file")
            print("\nStep 2: Extract")
            print("  Extract the downloaded zip to: data/GTSRB/")
            print("\nStep 3: Verify")
            print("  Run this script again and choose option 1 to verify")
            print("\nAlternatively, run:")
            print("  python utils/dataset_utils.py --action verify")
        
        elif choice == '3':
            # Train model
            if not check_dataset():
                print("\n❌ Dataset not found. Please download it first (option 2).")
                continue
            
            print_header("Model Training")
            print("\nChoose training mode:")
            print("1. Quick training (10 epochs) - ~15 minutes")
            print("2. Standard training (30 epochs) - ~45 minutes")
            print("3. Custom training")
            
            train_choice = input("\nChoice (1-3): ").strip()
            
            if train_choice == '1':
                run_command("python train_model.py --epochs 10", 
                           "Starting quick training")
            elif train_choice == '2':
                run_command("python train_model.py --epochs 30", 
                           "Starting standard training")
            elif train_choice == '3':
                epochs = input("Number of epochs (default 30): ").strip() or "30"
                batch = input("Batch size (default 32): ").strip() or "32"
                run_command(f"python train_model.py --epochs {epochs} --batch_size {batch}", 
                           "Starting custom training")
        
        elif choice == '4':
            # Test on image
            if not check_model():
                print("\n❌ Model not trained. Please train it first (option 3).")
                continue
            
            print_header("Test Model on Image")
            image_path = input("\nEnter image path (or press Enter for webcam test): ").strip()
            
            if image_path:
                run_command(f"python test_model.py --action image --image {image_path}",
                           "Testing model on image")
            else:
                run_command("python test_model.py --action webcam --duration 10",
                           "Testing model on webcam (10 seconds)")
        
        elif choice == '5':
            # Real-time detection
            if not check_model():
                print("\n❌ Model not trained. Please train it first (option 3).")
                continue
            
            print_header("Real-Time Detection")
            print("\nControls:")
            print("  Q - Quit")
            print("  S - Save screenshot")
            print("  P - Pause/Resume")
            print("\nStarting in 3 seconds...")
            
            import time
            time.sleep(3)
            
            run_command("python main_detector.py", 
                       "Starting real-time detection")
        
        elif choice == '6':
            # Dataset statistics
            if not check_dataset():
                print("\n❌ Dataset not found. Please download it first (option 2).")
                continue
            
            run_command("python utils/dataset_utils.py --action all",
                       "Generating dataset statistics")
        
        elif choice == '7':
            print("\n👋 Goodbye! Happy coding!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-7.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
