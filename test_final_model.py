from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import time
import json
from datetime import datetime

def ultimate_test():
    print(" ULTIMATE MODEL TESTING SUITE")
    print("=" * 50)
    
    model = YOLO("best.pt")
    
    test_images_path = "Hackathon2_test1/test1/images"
    test_images = list(Path(test_images_path).glob("*.*"))
    
    print(f" Test images: {len(test_images)}")
    
    print("\n 1. OFFICIAL VALIDATION (All Images)...")
    metrics = model.val(data="yolo_params.yaml", split="test", verbose=True)
    
    print("\n 2. BATCH PROCESSING ALL IMAGES...")
    output_dir = "final_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    start_time = time.time()
    
    for i, img_path in enumerate(test_images):
        results = model(str(img_path), conf=0.5, verbose=False)
        
        if i % 10 == 0 and len(results) > 0:
            annotated_img = results[0].plot()
            cv2.imwrite(f"{output_dir}/sample_{img_path.name}", annotated_img)
            processed_count += 1
    
    end_time = time.time()
    
    print("\n 3. SAVING COMPREHENSIVE RESULTS...")
    results_data = {
        "test_timestamp": datetime.now().isoformat(),
        "total_test_images": len(test_images),
        "performance_metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.p),
            "recall": float(metrics.box.r)
        },
        "inference_speed": f"{(end_time - start_time) / len(test_images) * 1000:.1f}ms per image",
        "samples_saved": processed_count,
        "class_performance": {}
    }
    
    for i, class_name in model.names.items():
        results_data["class_performance"][class_name] = {
            "mAP50": float(metrics.box.ap50[i]),
            "mAP50_95": float(metrics.box.ap[i])
        }
    
    with open("final_test_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print("\n ULTIMATE TESTING COMPLETED!")
    print(f" Official mAP50: {metrics.box.map50:.3f} ({metrics.box.map50*100:.1f}%)")
    print(f" Samples saved: {output_dir}/")
    print(f" Full results: final_test_results.json")

if __name__ == "__main__":
    ultimate_test()