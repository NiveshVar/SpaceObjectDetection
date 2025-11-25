# Space Station Safety Equipment Detection

##  Project Overview

This project implements a cutting-edge **AI-powered object detection system** specifically designed for identifying critical safety equipment in space station environments. Developed for the **Duality AI Space Station Hackathon**, our solution leverages **YOLO (You Only Look Once) architecture** and **synthetic data from Duality AI's Falcon digital twin platform** to create a robust, real-time safety monitoring system.

The system addresses the critical challenge of automated safety equipment monitoring in environments where human inspection is limited and response times are crucial for mission success and crew safety.

---

##  Key Features

###  Advanced Detection Capabilities
- **Real-time Object Detection**: 20ms inference time per image enabling live video processing
- **7 Safety Equipment Classes**: Comprehensive coverage of critical space station safety items
- **High Precision Operations**: 84.8% precision ensuring reliable detections
- **Multi-scale Detection**: Effective identification of both large and small safety equipment

###  User-Friendly Interface
- **Web Application**: Intuitive Gradio-based interface for easy interaction
- **Mobile Responsive**: Optimized for tablets, phones, and desktop systems
- **Real-time Visualization**: Instant bounding box annotations and confidence scores
- **Batch Processing**: Support for multiple image upload and processing

###  Technical Excellence
- **Model Performance**: 71.2% mAP@0.5, significantly exceeding the 40-50% benchmark
- **Continuous Learning**: Integration with Falcon digital twin for ongoing model improvements
- **Production Ready**: Scalable architecture suitable for deployment in actual space station systems
- **Comprehensive Analytics**: Detailed performance metrics and detection statistics

---

##  Performance Metrics

| Metric | Score | Benchmark | Status |
|--------|-------|-----------|---------|
| **mAP@0.5** | 71.2% | 40-50% |  **Exceeded** |
| **Precision** | 84.8% | >70% |  **Exceeded** |
| **Recall** | 63.0% | >50% |  **Exceeded** |
| **Inference Speed** | 20ms/image | <50ms |  **Exceeded** |
| **Model Size** | 22.5MB | <100MB |  **Optimized** |

### Class-wise Performance
| Safety Equipment | mAP@0.5 | Precision | Recall | Status |
|------------------|---------|-----------|--------|---------|
| **Oxygen Tank** | 81.8% | 89.2% | 74.7% |  Excellent |
| **Nitrogen Tank** | 75.3% | 88.7% | 65.0% |  Excellent |
| **First Aid Box** | 79.4% | 77.3% | 70.6% |  Excellent |
| **Fire Alarm** | 75.1% | 90.8% | 69.6% |  Excellent |
| **Safety Switch Panel** | 58.6% | 72.2% | 50.0% |  Good |
| **Emergency Phone** | 60.8% | 90.0% | 53.6% |  Good |
| **Fire Extinguisher** | 67.7% | 85.2% | 57.6% |  Good |

---

##  Installation & Setup

### Prerequisites

- **Python 3.8+** (Recommended: 3.10+)
- **CUDA-capable GPU** (Recommended for training, optional for inference)
- **8GB+ RAM** (16GB recommended for optimal performance)
- **5GB+ Free Disk Space** for models and datasets

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-username/space-station-safety-detection.git
cd space-station-safety-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (if available)
# Or train your own model using the training scripts
```

### Detailed Installation

#### 1. Environment Setup
```bash
# Using conda (alternative)
conda create -n space-safety python=3.10
conda activate space-safety
```

#### 2. Dependency Installation
```bash
# Core dependencies
pip install ultralytics torch torchvision opencv-python

# Web interface and utilities
pip install gradio pillow numpy matplotlib seaborn

# Data processing
pip install pandas pyyaml scikit-learn albumentations
```

#### 3. Model Setup
```bash
# For training from scratch
python scripts/train.py --epochs 50 --batch-size 16

# Or use pre-trained weights (when available)
# Place best.pt in models/ directory
```

---

##  Usage Guide

### Web Application
```bash
# Launch the web interface
python app.py

# Access the application at:
# Local: http://localhost:7860
# Public: Use the share link provided in console
```

### Command Line Inference
```bash
# Single image detection
python scripts/predict.py --source images/test.jpg --conf 0.5

# Batch processing
python scripts/predict.py --source images/ --conf 0.5 --save-txt

# Webcam real-time detection
python scripts/predict.py --source 0 --conf 0.6
```

### Model Training
```bash
# Basic training
python scripts/train.py --epochs 50 --batch-size 16 --img-size 640

# Advanced training with augmentation
python scripts/train.py --epochs 100 --batch-size 8 --img-size 640 --augment

# Resume training from checkpoint
python scripts/train.py --resume runs/detect/train/weights/last.pt
```

---

##  Project Structure

```
space-station-safety-detection/
├──  models/                          # Trained model weights
│   ├── best.pt                        # Best performing model
│   └── last.pt                        # Latest training checkpoint
├──  scripts/                        # Core functionality scripts
│   ├── train.py                       # Model training script
│   ├── predict.py                     # Inference and evaluation
│   ├── visualize.py                   # Results visualization
│   └── utils/                         # Utility functions
├──  config/                         # Configuration files
│   ├── yolo_params.yaml               # YOLO training configuration
│   └── app_config.py                  # Application settings
├──  data/                           # Dataset and processing
│   ├── train/                         # Training images and labels
│   ├── val/                           # Validation images and labels
│   └── test/                          # Test images and labels
├──  examples/                       # Sample images for demonstration
│   ├── space_station_1.jpg
│   ├── space_station_2.jpg
│   └── space_station_3.jpg
├──  runs/                           # Training outputs and results
│   ├── detect/train/                  # Training run results
│   └── detect/val/                    # Validation results
├── app.py                            # Main web application
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── LICENSE                           # Project license
```

---

##  Web Application Features

###  Image Upload & Processing
- **Drag & Drop Interface**: Intuitive file selection
- **Multiple Format Support**: JPEG, PNG, BMP, TIFF compatibility
- **Real-time Preview**: Immediate visual feedback
- **Batch Upload**: Process multiple images simultaneously

###  Detection Controls
- **Confidence Adjustment**: Dynamic threshold from 0.1 to 0.9
- **Class Filtering**: Selective detection of specific equipment types
- **Processing Modes**: Real-time vs. batch optimization
- **Export Options**: Multiple output format support

###  Results Visualization
- **Color-coded Bounding Boxes**: Distinct colors for each equipment class
- **Confidence Overlays**: Clear display of detection certainty
- **Interactive Elements**: Click bounding boxes for detailed information
- **Comparison Mode**: Side-by-side original vs. detected views

###  Analytics Dashboard
- **Real-time Metrics**: Live performance statistics
- **Detection History**: Timeline of processed images
- **Performance Charts**: Visual representation of model accuracy
- **Export Capabilities**: Save results in JSON, CSV, or image formats

---

##  Technical Architecture

### AI/ML Stack
- **Object Detection**: Ultralytics YOLOv8/YOLO11
- **Deep Learning Framework**: PyTorch 2.0+
- **Computer Vision**: OpenCV 4.8+, Albumentations
- **Model Optimization**: CUDA acceleration, mixed precision training

### Web Framework
- **Interface**: Gradio 3.0+
- **Backend**: FastAPI (optional)
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas

### Data Pipeline
- **Synthetic Generation**: Duality AI Falcon Platform
- **Data Augmentation**: Mosaic, MixUp, Color jittering
- **Validation**: Cross-validation, confusion matrix analysis
- **Quality Assurance**: Automated data integrity checks

---

##  Detection Classes

The system is trained to detect 7 critical safety equipment categories:

1. ** Oxygen Tank** - Life support systems
2. ** Nitrogen Tank** - Environmental control systems  
3. ** First Aid Box** - Medical emergency equipment
4. ** Fire Alarm** - Fire detection and alert systems
5. ** Safety Switch Panel** - Emergency control interfaces
6. ** Emergency Phone** - Critical communication devices
7. ** Fire Extinguisher** - Fire suppression equipment

---

##  Model Development

### Training Strategy
- **Transfer Learning**: Pre-trained on COCO dataset
- **Progressive Training**: Multi-stage fine-tuning approach
- **Data Augmentation**: Space-environment specific transformations
- **Hyperparameter Optimization**: Automated search for optimal settings

### Performance Optimization
- **Inference Speed**: Model quantization and optimization
- **Accuracy**: Advanced augmentation and training techniques
- **Robustness**: Testing under varied lighting and occlusion conditions
- **Scalability**: Support for multiple deployment scenarios

### Validation Approach
- **Cross-validation**: 5-fold validation on synthetic dataset
- **Real-world Testing**: Evaluation on space station simulation images
- **Performance Metrics**: Comprehensive evaluation using industry standards
- **A/B Testing**: Comparative analysis of different model architectures

---

##  Unique Features

###  Space-Environment Optimized
- **Lighting Adaptation**: Robust performance under variable lighting conditions
- **Occlusion Handling**: Effective detection of partially visible equipment
- **Angle Invariance**: Reliable detection from multiple camera perspectives
- **Reflection Management**: Special handling of metallic surface reflections

###  Continuous Improvement
- **Digital Twin Integration**: Seamless connection with Falcon platform
- **Automated Retraining**: Performance-triggered model updates
- **Feedback Loops**: Crew input integration for model refinement
- **Version Control**: Comprehensive model version management

###  Safety-First Design
- **High Reliability**: 84.8% precision for critical safety applications
- **Failure Modes**: Graceful degradation and error handling
- **Alert Systems**: Automated notification of detection issues
- **Audit Trails**: Complete processing history and decision logs

---

##  Future Enhancements

### Planned Improvements
- **Additional Equipment Classes**: Expand detection to 15+ safety items
- **3D Spatial Awareness**: Depth estimation and spatial mapping
- **Predictive Analytics**: Equipment usage patterns and maintenance forecasting
- **Multi-modal Detection**: Combine visual with sensor data fusion

### Integration Opportunities
- **Space Station Systems**: Direct integration with ISS and commercial stations
- **Mission Control**: Real-time data streaming to ground control
- **Crew Interfaces**: Astronaut wearable integration
- **Research Platforms**: Scientific data collection and analysis

---

##  Team & Acknowledgments

### Development Team
- **Lead Developer**: [Your Name/Team Name]
- **AI/ML Engineering**: [Team Members]
- **System Architecture**: [Team Members]
- **Testing & Validation**: [Team Members]

### Acknowledgments
- **Duality AI** for providing the Falcon digital twin platform
- **Ultralytics** for the YOLO framework and continuous improvements
- **NASA** for inspiration and space station safety standards
- **Open Source Community** for invaluable tools and libraries

### Hackathon Participation
This project was developed as part of the **Duality AI Space Station Hackathon**, focusing on advancing AI applications for space exploration and safety.

---

##  License

This project is developed for educational and research purposes as part of the Duality AI Hackathon. Please refer to the LICENSE file for detailed terms and conditions.

---

##  Contributing

While this is primarily a hackathon project, we welcome feedback and suggestions:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Ensure code follows project standards

For major changes, please open an issue first to discuss proposed modifications.

---


##  Conclusion

This Space Station Safety Equipment Detection system represents a significant step forward in automated safety monitoring for space environments. By combining state-of-the-art object detection with synthetic data generation, we've created a solution that not only meets but exceeds performance benchmarks while demonstrating the practical application of AI in critical safety scenarios.

**Ready to launch your space station safety monitoring?** 🚀

```bash
python app.py
```

*Begin detecting safety equipment in your space station images today!*
