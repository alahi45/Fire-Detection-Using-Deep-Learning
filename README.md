# 🔥 AI-Powered Fire Detection using Deep Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)

A real-time fire detection system built with a Convolutional Neural Network (CNN) using Transfer Learning. This project can accurately classify the presence of fire in images and video streams, serving as a powerful tool for automated safety monitoring.

![Project Demo GIF](./assets/demo_video.gif)
---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 📝 Project Overview

This repository contains the code for an AI-powered fire detection model. The primary goal is to leverage computer vision and deep learning to create a system that can instantly identify fires from a standard camera feed. By using a pre-trained model (MobileNetV2), this project achieves high accuracy, making it a viable proof-of-concept for modern fire safety solutions.

---

## 🎯 Problem Statement

Traditional fire detection systems like smoke and heat sensors are effective but require physical proximity to a fire, leading to potentially critical delays. This is especially true for large open areas or outdoor environments. The objective of this project is to overcome this limitation by creating a vision-based system that can detect fires from a distance, providing a faster, more versatile monitoring solution.

---

## ✨ Features

- **Image Classification:** Classifies any given image as containing 'Fire' or 'No Fire'.
- **Video Processing:** Analyzes video files frame-by-frame to detect fire in real-time.
- **High Accuracy:** Achieves over 98% accuracy on the validation set.
- **Robust Model:** Built using transfer learning to generalize well to new, unseen images.

---

## 🧠 Methodology

The model was developed following a standard data science workflow:

### 1. Dataset
The model was trained on a public dataset from Kaggle containing ~1,000 images, balanced between two classes:
- `fire_images`
- `non_fire_images`

**Data augmentation** (random rotations, zooms, flips) was heavily applied to the training set to improve the model's robustness and prevent overfitting.

### 2. Model Architecture
This project employs **transfer learning** with the **MobileNetV2** architecture, pre-trained on the ImageNet dataset. The core convolutional layers of MobileNetV2 were frozen, and custom classification layers were added on top. This approach leverages the powerful feature extraction capabilities of a state-of-the-art model and fine-tunes it for our specific task.

### 3. Training
The model was trained for 15 epochs using the Adam optimizer and binary cross-entropy loss function. The training was conducted in a Google Colab environment, leveraging its free GPU resources.

---

## 📊 Results

The model's performance was evaluated using several key metrics, demonstrating its effectiveness.

- **Accuracy:** >98%
- **AUC Score:** 0.99

#### Training History
The training and validation curves for accuracy and loss show stable learning with no significant overfitting.

![Training History](./assets/history.png)
#### Confusion Matrix
The confusion matrix highlights the model's excellent ability to distinguish between the two classes, with very few misclassifications.

![Confusion Matrix](./assets/matrix.png)
#### ROC Curve
An AUC score of 0.99 indicates an outstanding classification performance across all thresholds.

![ROC Curve](./assets/roc.png)
---

## 🚀 How to Run

You can replicate this project by running the provided Jupyter Notebook (`Fire_Detection.ipynb`) in Google Colab.

### Prerequisites
- Python 3.8+
- A Kaggle account and an API token (`kaggle.json`)

### Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow scikit-learn seaborn matplotlib pandas kaggle
    ```
3.  **Run in Google Colab:**
    - Open the `.ipynb` notebook in Google Colab.
    - The notebook will prompt you to upload your `kaggle.json` file to download the dataset directly.
    - Run the cells sequentially to prepare the data, train the model, and evaluate its performance.

---

## 💻 Technologies Used
- **Python**
- **TensorFlow & Keras** (for model building and training)
- **Scikit-learn** (for performance metrics like ROC, Confusion Matrix)
- **Seaborn & Matplotlib** (for data visualization)
- **NumPy & Pandas** (for data manipulation)
- **Google Colab** (for the development environment)

---

## 🔮 Future Work
- [ ] **Upgrade to Object Detection:** Implement a YOLO (You Only Look Once) model to draw bounding boxes around the exact location and size of the fire.
- [ ] **Real-Time Deployment:** Deploy the model on a low-cost edge device (like a Raspberry Pi with a camera) for a standalone, intelligent monitoring system.
- [ ] **Expand Functionality:** Add a smoke detection capability to the model.

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments
- The image dataset used for this project was provided by [phylake1337 on Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset).
