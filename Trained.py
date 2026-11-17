"""Train, evaluate and export a YOLOv8 crack detector.

Usage:
    python train.py --data /path/to/data.yaml --epochs 50
Outputs results.json (metrics + speed) and an ONNX model.
"""
import argparse
import json
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to YOLOv8-format data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base weights (n/s/m)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project="runs", name="crack")

    best = YOLO("runs/crack/weights/best.pt")
    m = best.val(data=args.data, imgsz=args.imgsz)

    results = {
        "model": args.model,
        "epochs": args.epochs,
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    best.export(format="onnx")  # used later for the quantization/edge project


if __name__ == "__main__":
    main()
