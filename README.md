# Melanoma Pipeline — Segmentation + Classification

End-to-end inference pipeline combining U-Net segmentation and EfficientNet-B0 classification for skin lesion analysis.

## How it works

Dermoscopic image
↓
U-Net (segmentation) → binary mask
↓
Overlay mask on original image
↓
EfficientNet-B0 (classification) → benign / malignant

## Stack
- PyTorch
- segmentation-models-pytorch (U-Net)
- EfficientNet-B0
- ISIC 2018 Task 1 (segmentation training data)
- Melanoma Skin Cancer Dataset 10k (classification training data)

## Models
- Segmentation: [Ai-Adam-Six-Sigma/melanoma-segmentation](https://huggingface.co/Ai-Adam-Six-Sigma/melanoma-segmentation)
- Classification: [Ai-Adam-Six-Sigma/melanoma-classifier](https://huggingface.co/Ai-Adam-Six-Sigma/melanoma-classifier)

## Project Structure
```plaintext
melanoma_pipeline_seg_class/
├── models/          # model weights (not in repo)
├── notebooks/
│   └── pipeline.ipynb
├── test_images/     # test images (not in repo)
└── source_code_src/ # source code (in progress)
```

## Related Projects
- [melanoma-classification](https://github.com/AdamSobanski/melanoma-classification)
- [melanoma-segmentation](https://github.com/AdamSobanski/melanoma-segmentation)

## ⚠️ Disclaimer
Educational prototype — not a diagnostic tool. Always consult a dermatologist.