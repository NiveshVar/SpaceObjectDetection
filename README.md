# 🚀 Space Station Safety Equipment Detection

## Project Overview
This project implements an AI-powered object detection system for identifying critical safety equipment in space station environments. Using YOLO (You Only Look Once) architecture and synthetic data from Duality AI's Falcon platform, the system detects 7 types of safety equipment in real-time.

## 🎯 Features
- **Real-time Object Detection**: 20ms inference time per image
- **7 Safety Equipment Classes**: OxygenTank, NitrogenTank, FirstAidBox, FireAlarm, SafetySwitchPanel, EmergencyPhone, FireExtinguisher
- **Web Application**: User-friendly interface for image upload and live detection
- **Performance Metrics**: 71.2% mAP, 84.8% precision on validation data
- **Continuous Learning**: Integration with Falcon digital twin for model improvements

## 📊 Performance
| Metric | Score | Benchmark |
|--------|-------|-----------|
| mAP@0.5 | 71.2% | 40-50% |
| Precision | 84.8% | >70% |
| Recall | 63.0% | >50% |
| Inference Speed | 20ms/image | <50ms |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/space-station-safety-detection.git
cd space-station-safety-detection

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
