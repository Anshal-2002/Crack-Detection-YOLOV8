# Crack Detection with YOLOv8

Detects cracks in concrete / road / wall images using YOLOv8.
Built to explore computer vision for civil infrastructure inspection.

## Results
Fill these in from `results.json` after training (use your real numbers only).

| Metric | Value |
|---|---|
| mAP@0.5 | [ ] |
| mAP@0.5:0.95 | [ ] |
| Precision / Recall | [ ] / [ ] |
| Inference speed | [ ] ms/image on [hardware] |

Add 2-3 sample prediction images here: `![sample](samples/pred1.jpg)`

## Dataset
[Dataset name, source link, license, number of train/val images]

## Setup
```bash
pip install -r requirements.txt
python train.py --data path/to/data.yaml --epochs 50
```

## Approach
- Model: YOLOv8n fine-tuned from COCO weights
- Image size 640, [epochs] epochs, batch [16], trained on [Colab T4 GPU]
- Exported to ONNX for edge deployment

## Limitations / Next steps
- [What failed: e.g. thin hairline cracks, shadows]
- Quantize to INT8 and benchmark on edge hardware
