# Space Station Safety Equipment Detection

## Project Overview

This project implements an AI-powered object detection system for identifying critical safety equipment in space station environments. Developed for the Duality AI Space Station Hackathon, our solution leverages YOLO (You Only Look Once) architecture and synthetic data from Duality AI's Falcon digital twin platform to create a robust, real-time safety monitoring system.

## Performance Summary

- Overall mAP50: 78.1% (Benchmark: 40-50%)
- Precision: 92.9%
- Recall: 70.5% 
- Inference Speed: 1.8ms per image (GPU)
- Model Size: 5.5MB
- Training Time: 3.25 hours

## Installation

### Prerequisites
- Python 3.8+
- 8GB RAM minimum
- 5GB free disk space

### Quick Setup
```bash
# Clone repository
git clone https://github.com/NiveshVar/Space-Station-Safety-Detection.git
cd Space-Station-Safety-Detection

# Install dependencies
pip install -r requirements.txt

# Run setup verification
python setup.py
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install ultralytics torch torchvision opencv-python gradio pillow numpy
```

## Usage

### Web Application
```bash
python app.py
```
Access the application at: http://127.0.0.1:7860

### Command Line Detection
```bash
# Single image
python predict.py --source image.jpg --conf 0.5

# Batch processing
python predict.py --source images/ --conf 0.5

# Webcam real-time
python predict.py --source 0 --conf 0.6
```

### Model Training
```bash
# Train from scratch
python train.py --epochs 50 --batch-size 16 --img-size 640

# Resume training
python train.py --resume best.pt --epochs 25
```

## Project Structure
```
Space-Station-Safety-Detection/
├── app.py                 # Web application
├── train.py              # Training script
├── predict.py            # Inference script
├── test_final_model.py   # Testing and validation
├── setup.py             # Setup verification
├── requirements.txt     # Dependencies
├── yolo_params.yaml    # Configuration
├── best.pt            # Trained model (78.1% mAP)
├── performance_report.txt # Performance summary
└── README.md           # Documentation
```

## Detection Classes
The system detects 7 critical safety equipment types:
1. Oxygen Tank
2. Nitrogen Tank
3. First Aid Box
4. Fire Alarm
5. Safety Switch Panel
6. Emergency Phone
7. Fire Extinguisher

## Model Architecture
- Base Model: YOLO11n
- Input Size: 640x640 pixels
- Parameters: 2.58 million
- Framework: PyTorch + Ultralytics
- Training Data: Synthetic from Duality AI Falcon

## Performance Details

### Overall Metrics
- mAP50: 78.1%
- Precision: 92.9%
- Recall: 70.5%
- F1-Score: 80.1%

### Class-wise Performance
- Oxygen Tank: 88.5% mAP
- Nitrogen Tank: 85.9% mAP
- First Aid Box: 85.1% mAP
- Fire Alarm: 75.9% mAP
- Safety Switch Panel: 72.7% mAP
- Emergency Phone: 62.4% mAP
- Fire Extinguisher: 76.4% mAP

## Technical Features

### Real-time Capabilities
- Live video processing support
- Webcam integration
- Batch image processing
- Multiple input formats (images, video, webcam)

### Web Application
- Drag-and-drop interface
- Confidence threshold adjustment
- Real-time visualization
- Results export functionality
- Mobile-responsive design

### Optimization
- Model quantization ready
- GPU acceleration support
- Efficient memory usage
- Fast inference times

## Testing and Validation

### Quick Test
```bash
python test_final_model.py
```

### Comprehensive Evaluation
The model has been validated on 1,408 test images with consistent performance across various lighting conditions and scenarios.

## Configuration

### Training Parameters (yolo_params.yaml)
```yaml
path: dataset_path
train: train/images
val: val/images
test: test/images
nc: 7
names: ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
```

## Deployment

### Local Deployment
```bash
python app.py --server-name 0.0.0.0 --server-port 7860
```

### Production Considerations
- GPU acceleration recommended for real-time processing
- Multiple camera feed support
- Integration with existing monitoring systems
- Continuous learning capabilities

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or image dimensions
2. **Model Loading Errors**: Verify PyTorch and Ultralytics versions
3. **Web Interface Access**: Use 127.0.0.1 instead of localhost

### Performance Tips
- Use GPU for faster inference
- Adjust confidence threshold based on use case
- Enable data augmentation for training
- Monitor system resources during processing

## Acknowledgments
- Duality AI for the Falcon digital twin platform
- Ultralytics for the YOLO framework
- NASA for space station safety standards inspiration

## License
This project was developed for the Duality AI Space Station Hackathon. The code is provided for educational and research purposes.

## Support
For technical issues or questions, please refer to the project documentation or create an issue in the repository.
