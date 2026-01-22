# DeepFake Video Detector
## A practical deep learning baseline for detecting manipulated video content using temporal feature learning.

Deepfake videos pose a growing risk to digital trust, enabling misinformation, identity misuse, and media manipulation at scale.
Detecting such content requires models that go beyond individual frames and instead analyze temporal inconsistencies across video sequences.

This project implements an deepfake detection pipeline that converts raw videos into fixed-length clips and applies a 3D Convolutional Neural Network to learn spatio-temporal patterns indicative of manipulation.
The system is designed with a production-aware approach, emphasizing modular code structure, reproducible model versioning, and portable model export (ONNX) for downstream deployment.

## Dataset
Due to storage and size constraints, the dataset is not included in this repository.
The original data consists of deepfake videos, which are converted into short clips before training.
The video-to-clip preprocessing logic is implemented in the training notebook, while this repository focuses on:

1. Loading preprocessed clips
2. Model architecture and training
3. Evaluation logic

## Dataset Location & structure

* Storage: Google Drive
* Path: MyDrive/DeepFake_Video_Detector

- Structure:
  - Processed(Video-to-clip)/
  - raw/
  - testing/


📁 Folder Architecture
```
DeepFake_Video_Detector/
│
├── src/
│   ├── preprocess.py      # Video → clip sampling
│   ├── dataset.py         # Dataset loading from saved clips
│   ├── model.py           # 3D CNN architecture
│   ├── train.py           # Training pipeline & ONNX export
│   └── evaluation.py      # Metrics & visualization
│
├── notebooks/
│   └── deepfake_cnn_training.ipynb
│
├── model/
│   └── deepfake_3dcnn_final1.onnx.dvc
│
├── .dvc/
├── .gitignore
├── .dvcignore
└── README.md
```

## Training Pipeline

1. Clips loaded using a custom Dataset
2. Data split into train / validation
3. Loss: Weighted CrossEntropy
4. Optimizer: Adam
5. Scheduler: StepLR
6. Training logic fully modularized in src/train.py

## Evaluation

* Evaluation includes:
* Validation accuracy
* Classification report
* Confusion matrix
* ROC curve (AUC)
* Precision–Recall curve

All evaluation logic is implemented in src/evaluation.py.

## Model Versioning (DVC)

The trained ONNX model is tracked using DVC
Only metadata is tracked in Git, keeping the repository lightweight while ensuring reproducibility.

## Techstack

* Python
* PyTorch
* OpenCV
* NumPy
* scikit-learn
* DVC
* Google Colab(GPU)
* VS CODE
* git



## Future Scope

Larger backbones

Temporal attention

Audio–visual modeling

Frame-level localization

## Author
### Indrayani Parande
#### Data Scientist| Deep Learning & Computer Vision