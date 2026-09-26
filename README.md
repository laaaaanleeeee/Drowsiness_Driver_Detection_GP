# Driver Drowsiness Detection System

A real-time driver drowsiness detection and warning system using computer vision and deep learning.

## Overview

This project detects visual signs of driver drowsiness from camera input and provides real-time warnings. Three independent approaches were implemented and evaluated:

- **Method 1:** MediaPipe Face Mesh + EAR/MAR rule-based detection
- **Method 2:** YOLO object detection for `eyes_closed` and `yawning`
- **Method 3:** YOLO classification for eye and mouth states

## System Architecture

```text
               Camera Input
                    │
                    ▼
     ├── Method 1: MediaPipe + EAR/MAR
     │
     ├── Method 2: YOLOv8s Detection
     │
     └── Method 3: YOLO Classification
                    │
                    ▼
             Temporal Analysis
                    │
                    ▼
             Warning / Alert
```

## Methods

### 1. MediaPipe + EAR/MAR

MediaPipe Face Mesh is used to extract facial landmarks. Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) are calculated to detect prolonged eye closure and yawning.

### 2. YOLO Object Detection

A fine-tuned YOLOv8 model detects:

- `eyes_closed`
- `yawning`

Temporal conditions are applied to reduce false alarms in real-time detection.

### 3. YOLO Classification

Separate classification models are used for:

- **Eye:** `Open_Eyes` / `Closed_Eyes`
- **Mouth:** `No_yawn` / `Yawn`

The classification results are combined with temporal logic to determine drowsiness.

## Dataset

Datasets were prepared and processed for the corresponding detection and classification tasks.

The original datasets are not included in this repository where redistribution is restricted. Please refer to the original dataset sources and their licenses.

## Training

Training notebooks and experiment outputs are available in the repository and Google Colab.

- [Fine-tune YOLOv8 Detection – Colab](https://colab.research.google.com/drive/1onifbuY03OvC3LoK1n4vMLzFGphf5EGL?usp=sharing)
- [Fine-tune YOLOv8 Classification Eyes – Colab](https://colab.research.google.com/drive/1WMr4fyI_cgSVkBUvoAahLsMdAMVYXw3n?usp=sharing)
- [Fine-tune YOLOv8 Classification Mouth – Colab](https://colab.research.google.com/drive/1L_jHlO5z45DMi7Jo5TpiKiLXLw2xkTxN?usp=sharing)

## Results

| Method | Model | Task |
|---|---|---|
| Method 1 | MediaPipe Face Mesh | Rule-based detection |
| Method 2 | YOLO | Object detection |
| Method 3 | YOLO classification | Image classification |

Detailed training metrics, plots, and evaluation results are available in each method's `results/` directory.

## Demo

The system supports real-time camera input and generates an alert when drowsiness conditions are detected.

Demo images/videos are available in the `demo/` directory.

## Limitations

- Performance can be affected by lighting conditions and face visibility.
- Fixed thresholds may not generalize equally to every driver.
- Real-world performance may differ from offline dataset evaluation.

## Future Improvements

- Personalized drowsiness thresholds
- PERCLOS-based analysis
- Improved low-light robustness
- Edge/IoT deployment
- More advanced temporal modeling

## Author

**Le Hai Lan**

Graduation Project – Driver Drowsiness Detection System  
2025–2026
