# README for **Smart Plant Disease Detector**

![Smart Plant Disease Detector](https://github.com/FerasAlshash/Planet-Diseas/blob/main/SEdmkmBLlPZhdSdkqtLYm.jpg)

---

## 🌱 Project Overview
The **Smart Plant Disease Detector** is a web-based application built using Flask and TensorFlow that allows users to upload an image of a plant leaf and get predictions about potential diseases affecting the plant. The model uses a Convolutional Neural Network (CNN) trained on a dataset of various plant diseases to classify images into specific categories.

This project aims to assist farmers, researchers, and enthusiasts in identifying plant diseases early, enabling timely intervention and improving crop health.

---

## ⚙️ Features
- **Image Upload**: Users can upload an image of a plant leaf.
- **Real-time Prediction**: The application processes the uploaded image and predicts the disease with confidence scores.
- **User-friendly Interface**: A simple and intuitive HTML interface for seamless interaction.
- **High Accuracy**: Powered by a pre-trained CNN model with high accuracy on common plant diseases.

---

## 📋 Requirements
To run this project locally, ensure you have the following dependencies installed:

1. **Python 3.8+**
2. **Flask** - For building the web application.
3. **TensorFlow** - For loading and running the pre-trained CNN model.
4. **NumPy** - For numerical operations.
5. **Pillow (PIL)** - For image processing.

You can install the required Python packages using `pip`:
```bash
pip install flask tensorflow numpy pillow
