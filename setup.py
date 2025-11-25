"""
Space Station Safety Detection - Setup Script
Run this to quickly set up the project
"""

import os
import subprocess
import sys

def setup_project():
    print(" Setting up Space Station Safety Detection...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(" Python 3.8+ required")
        return False
    
    print(" Python version check passed")
    
    print(" Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Dependencies installed successfully")
    except Exception as e:
        print(f" Failed to install dependencies: {e}")
        return False
    
    # Verify model file
    if not os.path.exists("best.pt"):
        print("  Model file 'best.pt' not found")
        print("   Please ensure the trained model is in the project root")
    else:
        print(" Model file found")
    
    # Create necessary directories
    directories = ["examples", "test_results", "quick_test_results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Created directory: {directory}")
    
    print("\n Setup completed successfully!")
    print("\n Next steps:")
    print("1. Run: python test_final_model.py (to verify installation)")
    print("2. Run: python app.py (to start web application)")
    print("3. Open http://127.0.0.1:7860 in your browser")
    
    return True

if __name__ == "__main__":
    setup_project()
