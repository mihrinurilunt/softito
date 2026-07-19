# Deep Learning Portfolio

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]() [![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)]() [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)]() [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)]()

This folder contains independent, portfolio-quality deep learning notebooks. Each notebook is designed as a mini project that can be read and evaluated on its own.

> Goal: show practical ability across CNN image classification, transfer learning, object detection, semantic segmentation, and OCR/sequence recognition.

---

## Portfolio Projects

| Project | Notebook | Dataset | Main Skills |
|---|---|---|---|
| CNN Image Classification | [cnn_image_classification.ipynb](01-cnn-image-classification/cnn_image_classification.ipynb) | Intel Image Classification | CNN from scratch, preprocessing, evaluation, confusion matrix |
| Transfer Learning | [transfer_learning_resnet50.ipynb](02-transfer-learning-resnet50/transfer_learning_resnet50.ipynb) | Oxford Flowers 102 | ResNet50, feature extraction, fine-tuning, Grad-CAM |
| YOLO Object Detection | [yolo_object_detection.ipynb](03-yolo-object-detection/yolo_object_detection.ipynb) | COCO8 | YOLOv8, bounding boxes, IoU, NMS, mAP |
| Image Segmentation | [image_segmentation_unet.ipynb](04-image-segmentation-unet/image_segmentation_unet.ipynb) | Oxford-IIIT Pet Segmentation | U-Net, masks, Dice, IoU |
| OCR / CRNN | [crnn_text_recognition.ipynb](05-crnn-text-recognition/crnn_text_recognition.ipynb) | Synthetic License Plates | OCR, CRNN, recurrent layers, CTC loss |

---

## Repository Structure

```text
deep_learning/
├─ README.md
├─ requirements.txt
├─ 01-cnn-image-classification/
│  └─ cnn_image_classification.ipynb
├─ 02-transfer-learning-resnet50/
│  └─ transfer_learning_resnet50.ipynb
├─ 03-yolo-object-detection/
│  └─ yolo_object_detection.ipynb
├─ 04-image-segmentation-unet/
│  └─ image_segmentation_unet.ipynb
├─ 05-crnn-text-recognition/
│  └─ crnn_text_recognition.ipynb
└─ assets/
```

---

## What This Portfolio Demonstrates

- Building and evaluating a CNN from scratch
- Using transfer learning with a modern pretrained backbone
- Training and validating an object detector
- Understanding semantic segmentation with U-Net
- Building an OCR-style sequence recognition model with CRNN and CTC loss
- Communicating results with clean Markdown, diagrams, plots, and practical analysis

---

## How to Run

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open any notebook and run it from top to bottom.

Some notebooks download public datasets during execution. The YOLO notebook also downloads pretrained weights and COCO8 through Ultralytics.

---

## Notes

- Each notebook includes a table of contents, visualizations, key learnings, common mistakes, interview questions, and further reading.
- Random seeds are set for reproducibility.
- The notebooks are intentionally independent; they do not require finishing one project before starting another.
