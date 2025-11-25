import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime

# Import YOLO - will be functional when model is available
try:
    from ultralytics import YOLO
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False
    print("⚠️ Ultralytics not available - running in demo mode")

class SafetyEquipmentDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model"""
        try:
            # Try to load the model - will work after training completes
            model_path = "models/best.pt"
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                self.model_loaded = True
                print("✅ Model loaded successfully!")
            else:
                print("⚠️ Model not found - running in demo mode")
                self.model_loaded = False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def detect_objects(self, image, confidence_threshold=0.5):
        """Detect safety equipment in the image"""
        if not self.model_loaded or self.model is None:
            # Demo mode - return sample detections
            return self.demo_detection(image)
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=confidence_threshold, imgsz=640)
            
            if len(results) == 0:
                return image, "No objects detected", []
            
            # Get annotated image
            annotated_image = results[0].plot()
            
            # Extract detection information
            detections = []
            detection_data = []
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append(f"{class_name}: {confidence:.2f}")
                detection_data.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })
            
            # Create detection summary
            if detections:
                summary = f"🚀 Detected {len(detections)} safety equipment items:\n" + "\n".join(detections)
            else:
                summary = "No safety equipment detected"
            
            return annotated_image, summary, detection_data
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return image, f"Error during detection: {str(e)}", []

    def demo_detection(self, image):
        """Provide demo detection when model is not available"""
        # Create a simple demo output
        demo_image = image.copy()
        
        # Add some sample bounding boxes for demo
        height, width = demo_image.shape[:2]
        
        # Sample bounding boxes for demo
        demo_boxes = [
            {"class": "OxygenTank", "color": (0, 255, 0), "bbox": [width//4, height//4, width//2, height//2]},
            {"class": "FireExtinguisher", "color": (0, 0, 255), "bbox": [width//3, height//3, 2*width//3, 2*height//3]}
        ]
        
        for box in demo_boxes:
            x1, y1, x2, y2 = box["bbox"]
            cv2.rectangle(demo_image, (x1, y1), (x2, y2), box["color"], 2)
            cv2.putText(demo_image, box["class"], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box["color"], 2)
        
        demo_summary = "🔧 DEMO MODE\nOxygenTank: 0.87\nFireExtinguisher: 0.92\n\n(Actual detection requires trained model)"
        
        return demo_image, demo_summary, []

# Initialize detector
detector = SafetyEquipmentDetector()

def process_image(input_image, confidence_threshold):
    """Process image through the detection pipeline"""
    if input_image is None:
        return None, "Please upload an image", None
    
    try:
        # Convert Gradio image to OpenCV format
        if isinstance(input_image, str):
            image = cv2.imread(input_image)
        else:
            image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        # Run detection
        annotated_image, summary, detection_data = detector.detect_objects(
            image, confidence_threshold
        )
        
        # Convert back to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Create JSON output
        json_output = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "detections": detection_data,
            "summary": summary,
            "confidence_threshold": confidence_threshold
        }, indent=2)
        
        return annotated_image_rgb, summary, json_output
        
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        return input_image, error_msg, None

def update_confidence_display(confidence):
    """Update confidence threshold display"""
    return f"Confidence Threshold: {confidence}"

# Create the web interface
with gr.Blocks(
    title="Space Station Safety Equipment Detector",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .detection-summary {
        background: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🚀 Space Station Safety Equipment Detection
        
        **AI-powered object detection for space station safety monitoring**
        
        This system detects 7 critical safety equipment items in space station environments using 
        YOLO object detection trained on synthetic data from Duality AI's Falcon platform.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            
            input_image = gr.Image(
                label="Space Station Image",
                type="numpy",
                height=300,
                sources=["upload", "clipboard"]
            )
            
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.1,
                label="Confidence Threshold",
                info="Higher values = more confident detections"
            )
            
            confidence_display = gr.Textbox(
                label="Current Setting",
                value="Confidence Threshold: 0.5",
                interactive=False
            )
            
            detect_btn = gr.Button(
                "🔍 Detect Safety Equipment",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ### 🎯 Detection Classes
            - 🟦 Oxygen Tank
            - 🟪 Nitrogen Tank  
            - 🟩 First Aid Box
            - 🟥 Fire Alarm
            - 🟨 Safety Switch Panel
            - 🟧 Emergency Phone
            - 🔴 Fire Extinguisher
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            
            output_image = gr.Image(
                label="Annotated Image",
                height=400,
                interactive=False
            )
            
            results_summary = gr.Textbox(
                label="Detection Summary",
                lines=4,
                max_lines=6,
                interactive=False,
                elem_classes="detection-summary"
            )
            
            json_output = gr.JSON(
                label="Detailed Results (JSON)",
                visible=True
            )
            
            with gr.Row():
                clear_btn = gr.Button("🔄 Clear", variant="secondary")
                export_btn = gr.Button("💾 Export Results", variant="secondary")
    
    # Examples section
    gr.Markdown("### 🖼️ Example Images")
    gr.Examples(
        examples=[
            ["examples/space_station_1.jpg"],
            ["examples/space_station_2.jpg"],
            ["examples/space_station_3.jpg"]
        ],
        inputs=input_image,
        outputs=[output_image, results_summary, json_output],
        fn=process_image,
        cache_examples=False,
        label="Try these example images:"
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Technical Details**: YOLO Object Detection • 71.2% mAP • 20ms inference • 7 safety equipment classes
        
        Developed for Duality AI Space Station Hackathon
        """
    )
    
    # Event handlers
    confidence_slider.change(
        fn=update_confidence_display,
        inputs=confidence_slider,
        outputs=confidence_display
    )
    
    detect_btn.click(
        fn=process_image,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, results_summary, json_output]
    )
    
    clear_btn.click(
        fn=lambda: [None, "Upload an image to begin detection", None],
        inputs=[],
        outputs=[input_image, results_summary, json_output]
    )

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    
    print("🚀 Starting Space Station Safety Detection App...")
    print("📧 App will be available at: http://localhost:7860")
    print("🔗 For public sharing, use the share link that appears after startup")
    
    demo.launch(
        server_name="0.0.0.0",
        share=False,  # Set to True for public sharing
        debug=True
    )
