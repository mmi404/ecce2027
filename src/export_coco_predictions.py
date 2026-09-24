"""
Export raw predictions from a trained YOLO model to COCO-format JSON at conf=0.001.
Designed for offline evaluation without GPU.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

TRAIN_CLASSES = [
    'auto_rickshaw', 'bicycle', 'bus', 'car', 'cart_vehicle', 
    'construction_vehicle', 'motorbike', 'person', 'priority_vehicle', 
    'three_wheeler', 'truck'
]

def export_predictions(
    weights_path,
    manifest_path,
    output_json_path,
    conf_thresh=0.001,
    iou_thresh=0.7,
    imgsz=640,
    device=0,
    batch=16
):
    print(f"=== Exporting COCO Predictions ===")
    print(f"Model weights:     {weights_path}")
    print(f"Target manifest:   {manifest_path}")
    print(f"Output JSON:       {output_json_path}")
    print(f"Confidence cutoff: {conf_thresh}")
    print(f"NMS IoU cutoff:    {iou_thresh}")
    print(f"Image resolution:  {imgsz}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
    with open(manifest_path, 'r') as f:
        image_paths = [l.strip() for l in f if l.strip()]
        
    print(f"Total images to evaluate: {len(image_paths)}")
    
    model = YOLO(weights_path)
    
    coco_predictions = []
    class_counts = {c: 0 for c in TRAIN_CLASSES}
    
    t0 = time.time()
    # Run prediction in batches using stream=True for memory efficiency
    results = model.predict(
        source=image_paths,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        device=device,
        batch=batch,
        stream=True,
        verbose=False
    )
    
    total_boxes = 0
    for idx, r in enumerate(results):
        img_file = r.path
        img_id = Path(img_file).stem
        
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
            
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            w = x2 - x1
            h = y2 - y1
            conf = float(confs[i])
            cls_id = int(clss[i])
            
            coco_predictions.append({
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(w), 2), round(float(h), 2)],
                "score": round(conf, 5)
            })
            total_boxes += 1
            if 0 <= cls_id < len(TRAIN_CLASSES):
                class_counts[TRAIN_CLASSES[cls_id]] += 1
                
        if (idx + 1) % 200 == 0 or (idx + 1) == len(image_paths):
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"Processed {idx + 1}/{len(image_paths)} images ({fps:.1f} img/s), {total_boxes} detections...")
            
    total_time = time.time() - t0
    print(f"\nInference completed in {total_time:.2f}s ({len(image_paths)/total_time:.1f} img/s).")
    print(f"Total detections at conf>={conf_thresh}: {total_boxes}")
    
    print("\nDetections per class:")
    for cname, cnt in class_counts.items():
        print(f"  {cname:22s}: {cnt}")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_predictions, f, indent=2)
        
    file_size_mb = os.path.getsize(output_json_path) / (1024 * 1024)
    print(f"\nSuccessfully wrote COCO predictions to: {output_json_path} ({file_size_mb:.2f} MB)")
    
    # Verification check
    with open(output_json_path, 'r') as f:
        loaded = json.load(f)
    assert len(loaded) == total_boxes, f"Mismatch: expected {total_boxes}, loaded {len(loaded)}"
    print(f"Verified: valid JSON with {len(loaded)} records.")
    return output_json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO predictions to COCO JSON at conf=0.001")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained last.pt weights")
    parser.add_argument("--manifest", type=str, required=True, help="Path to image manifest txt")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="Device (0 or cpu)")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()
    
    export_predictions(
        weights_path=args.weights,
        manifest_path=args.manifest,
        output_json_path=args.output,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch
    )
