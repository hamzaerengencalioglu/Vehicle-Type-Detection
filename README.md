# 🚗 Vehicle Classification Desktop Application

This is a modern, AI-powered desktop application that classifies the type of vehicle in an image using a deep learning model. Built with **PyTorch**, **CustomTkinter**, and **ResNet34**, it offers an intuitive and responsive GUI for real-time image classification.

---

## 📌 Overview

This application allows users to upload an image and automatically detects whether the image contains one of the following vehicle types:

- **Bike**
- **Car**
- **Motorcycle**
- **Plane**
- **Ship**
- **Train**

With a focus on **modern design**, **accessibility**, and **efficiency**, this tool can be used for educational purposes, AI/ML demonstrations, or as a base for production-level vehicle detection systems.

---

## 🚀 Features

- 🔍 **Real-time vehicle prediction** using a fine-tuned ResNet34 model
- 🖼️ **Image upload support** for `.jpg`, `.jpeg`, `.png` formats
- 🌙 **Dark mode UI** with accent highlights
- 🌀 **Loading animation** during prediction
- 💻 **Fullscreen experience** for clean and distraction-free usage
- ⚡ **Fast, optimized inference** with PyTorch
- ❌ Easy exit via `ESC` key or exit button

---

## 🧠 Model Architecture

- **Base Model**: `ResNet34` from `torchvision.models`
- **Pretrained on**: ImageNet
- **Fine-tuned**: On a 6-class custom vehicle dataset
- **Output Layer**: Custom fully-connected layer with 6 outputs
- **Classes**: `['Bike', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train']`

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-classification-app.git
cd vehicle-classification-app
