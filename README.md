# melanoma_pipline_yolo_seg_class_v2

End-to-end melanoma analysis pipeline that chains three independently trained models into a single inference flow: **YOLOv8n** for lesion detection, **U-Net (EfficientNet-B0 encoder)** for lesion segmentation, and **EfficientNet-B4** for malignant/benign classification.

Given a raw dermoscopic image, the pipeline locates the skin lesion, segments it, and classifies it — returning a label, a confidence score, and a 4-panel visualization (detection box → cropped lesion → segmentation mask → classification result).

## Pipeline overview

```
Input image
    │
    ▼
[1] YOLOv8n            → detects the lesion, picks the bbox with the highest confidence
    │
    ▼
[2] Crop               → crops the image to the detected bounding box
    │
    ▼
[3] U-Net (EffNet-B0)   → segments the lesion within the crop (binary mask)
    │
    ▼
[4] EfficientNet-B4     → classifies the crop as malignant / benign (sigmoid, 1 neuron)
    │
    ▼
Output: label, confidence, 4-panel visualization
```

## What's new in v2

- **Best-confidence bbox selection** — YOLO output previously took the first box returned; now the box with the highest `conf` is selected.
- **Consistent device handling** — all three models and their tensors are explicitly moved to `device` (CPU/GPU), preparing the pipeline for GPU inference and future ONNX/TensorRT export.
- **Classifier upgraded from B0 to B4** — swapped the `torchvision` EfficientNet-B0 (2-class softmax, 224×224 input) for a `timm` EfficientNet-B4 (1-neuron sigmoid, 380×380 input), improving malignant recall from **84.4% → 93.9%**.

## Models used

| Stage | Model | Details |
|---|---|---|
| Detection | YOLOv8n | mAP50 = 0.977 — [`melanoma_yolo`](https://github.com/AdamSobanski/melanoma_yolo) |
| Segmentation | U-Net (EfficientNet-B0 encoder) | Dice ≈ 0.88–0.89, trained on ISIC 2018 Task 1 — [`melanoma-segmentation`](https://github.com/AdamSobanski/melanoma-segmentation) |
| Classification | EfficientNet-B4 (timm) | Malignant recall 93.9%, trained on ISIC 2019+2020 — [`melanoma_classification_2`](https://github.com/AdamSobanski/melanoma_classification_2) |

Trained model weights and cards are also published on Hugging Face: [`Ai-Adam-Six-Sigma`](https://huggingface.co/Ai-Adam-Six-Sigma).

## Requirements

```
torch
torchvision
segmentation-models-pytorch
timm
ultralytics
Pillow
numpy
matplotlib
opencv-python
```

## Model weights

Place the trained weights in `../models/` relative to the notebook:

```
../models/yolo_best.pt
../models/unet_melanoma.pth
../models/efficientnet_b4_melanoma_final.pt
```

Weights can be downloaded from the [Hugging Face profile](https://huggingface.co/Ai-Adam-Six-Sigma) or trained from scratch using the individual repos linked above.

## Usage

**Single image:**

```python
test_img_path = r'../test_image/ISIC_0012603.jpg'
label, confidence = run_pipeline(test_img_path)

if label is not None:
    print(f'Result: {label} ({confidence:.2%})')
else:
    print('No detection from YOLO.')
```

**Batch over a folder:**

```python
test_folder = r'../test_images'
# iterates over all .jpg/.jpeg/.png files in the folder,
# runs the full pipeline on each, and prints a summary of predictions
```

If YOLO detects no lesion in the image, the pipeline returns `(None, None)` and no crop/mask/classification is attempted.

## Output

For each processed image the pipeline produces:
- **Predicted label** — `malignant` or `benign`
- **Confidence** — probability associated with the predicted label
- **4-panel visualization** — detection box, cropped lesion, segmentation mask, classification result with confidence

## Related repositories

- [`melanoma_yolo`](https://github.com/AdamSobanski/melanoma_yolo) — YOLOv8n lesion detection model
- [`melanoma-segmentation`](https://github.com/AdamSobanski/melanoma-segmentation) — U-Net lesion segmentation model
- [`melanoma_classification_2`](https://github.com/AdamSobanski/melanoma_classification_2) — EfficientNet-B4 malignant/benign classifier

## Notes

- Device (CPU/GPU) is detected automatically via `torch.cuda.is_available()` and used consistently across all three models.
- This repository focuses on **inference** (combining the three trained models). Training code and dataset details for each stage live in the respective repos linked above.
