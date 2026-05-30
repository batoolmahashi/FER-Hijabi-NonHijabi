# Deep Learning-Based Facial Expression Recognition for Hijabi and Non-Hijabi Faces

This repository contains the code for our paper presented at the **3rd International Conference on Emerging Trends and Applications in Artificial Intelligence (ICETAI 2026)**, Istanbul, Turkey.

## Paper

**Title:** Deep Learning-Based Facial Expression Recognition for Hijabi and Non-Hijabi Faces

**Authors:** [Batool Mahashi](https://github.com/batoolmahashi), [Samira Aldamen](https://github.com/samira-aldamen), [Wajd Almadi](https://github.com/wajdalmadi), [Shatha Arar](https://github.com/shathaarar), Rasha Obeidat, Farah AlShanik

**Institution:** Jordan University of Science and Technology, Irbid, Jordan

---

## Overview

Facial Expression Recognition (FER) systems have shown strong performance in recent years, but they often fail to perform equally across demographic groups. This work addresses the fairness gap between **hijabi and non-hijabi women** in FER systems.

We conduct a comprehensive evaluation of fairness-oriented training strategies using a **self-collected, balanced dataset** of 2,500 facial images across five emotion categories.

**Key finding:** Careful dataset design reduces fairness gaps more effectively than algorithmic interventions alone.

---

## Repository Structure

```
├── README.md
├── fairness_methods.ipynb           # Main notebook: Baselines + Fairness Methods
├── cross_dataset_evaluation.ipynb   # Cross-dataset evaluation (non-hijabi → hijabi)
├── gan_augmentation.ipynb           # GAN-based augmentation experiments (exploratory)
└── realtime_demo.py                 # Real-time emotion detection demo
```

---

## Dataset

We constructed a self-collected dataset of **2,500 facial images**:

- 1,250 hijabi women
- 1,250 non-hijabi women
- 5 emotion categories: Happy, Sad, Angry, Surprised, Neutral
- 250 images per emotion per group

All images were manually collected from publicly available websites.

**Dataset available on Kaggle:**
[Hijabi and Non-Hijabi Facial Expression Dataset](https://www.kaggle.com/datasets/batoolmahashi/hijabi-and-non-hijabi-facial-expression-dataset)

---

## Experiments

### 1. Architecture Comparison (`fairness_methods.ipynb`)
Comparison of four CNN backbones: ResNet-18, VGG-16, MobileNetV2, DenseNet-121.
ResNet-18 was selected as the backbone for all fairness experiments.

### 2. Fairness-Oriented Training Methods (`fairness_methods.ipynb`)

| Method | Overall Acc | Fairness Gap | Worst-Group Acc |
|--------|-------------|--------------|-----------------|
| Baseline (ResNet-18) | 95.48% | 0.48% | 91.89% |
| Balanced Sampling | 93.62% | 2.06% | 78.95% |
| Group-Weighted Loss | 95.74% | 1.02% | 89.47% |
| DANN | 96.81% | 2.16% | 89.19% |

### 3. Cross-Dataset Evaluation (`cross_dataset_evaluation.ipynb`)
A model trained exclusively on non-hijabi images was tested on hijabi faces, achieving only **27.04% accuracy** — highlighting the importance of inclusive data collection.

### 4. GAN-Based Augmentation (`gan_augmentation.ipynb`)
Exploratory study using **DCGAN** and **StyleGAN2 with R1 regularization** to generate synthetic hijabi facial expression images.

> Note: GAN experiments were conducted prior to the full dataset collection pipeline, using hijabi images only. Generated images showed promising coarse features but lacked fine emotional detail and were not used as substitutes for real data.

### 5. Real-Time Demo (`realtime_demo.py`)
A real-time proof-of-concept system built to validate the practical deployability of the trained models. The system uses a webcam and OpenCV's Haar Cascade classifier for face detection, applies the same preprocessing pipeline used during training, and runs the baseline model to predict emotions on each detected face in real time.

> Note: The pre-trained model file (`best_baseline_model.pth`) is not included in this repository due to file size limitations. You can generate it by running `fairness_methods.ipynb`.

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
tqdm
opencv-python
```

Install with:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow tqdm opencv-python
```

> All notebooks were developed and run on **Google Colab** with GPU support.

---

## Key Findings

- High overall accuracy does **not** guarantee fairness across demographic groups.
- The baseline ResNet-18 model trained on balanced, controlled data achieved the **smallest fairness gap (0.48%)** and **best worst-group accuracy (91.89%)**.
- Fairness gaps in FER are primarily driven by **domain mismatch and data characteristics**, not inherent algorithmic bias.
- **Better data beats better algorithms** when it comes to fairness under controlled conditions.


