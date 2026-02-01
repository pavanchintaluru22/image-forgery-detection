# Image Forgery Detection using Deep Learning and Grad-CAM

This project detects whether an image is **AUTHENTIC** or **FORGED** using a deep learning model and explains the decision using **Grad-CAM visualizations**.

The system is designed to work on **laptops and mobile devices via a server-based deployment**, making it lightweight and practical for real-world use.

---

## 🔍 Project Features

- Binary classification: **Authentic vs Forged**
- CNN-based deep learning model (TensorFlow / Keras)
- Laplacian preprocessing to highlight forgery artifacts
- **Grad-CAM** for visual explanation of model decisions
- Confidence score and decision margin for transparency
- Designed for future **FastAPI-based web/mobile deployment**

---

## 🗂️ Project Structure

# Image Forgery Detection using Deep Learning and Grad-CAM

This project detects whether an image is **AUTHENTIC** or **FORGED** using a deep learning model and explains the decision using **Grad-CAM visualizations**.

The system is designed to work on **laptops and mobile devices via a server-based deployment**, making it lightweight and practical for real-world use.

---

## 🔍 Project Features

- Binary classification: **Authentic vs Forged**
- CNN-based deep learning model (TensorFlow / Keras)
- Laplacian preprocessing to highlight forgery artifacts
- **Grad-CAM** for visual explanation of model decisions
- Confidence score and decision margin for transparency
- Designed for future **FastAPI-based web/mobile deployment**

---

## 🗂️ Project Structure

PythonProject1/
│
├── dataset/
│ ├── authentic/ # Genuine images
│ └── forged/ # Forged images
│
├── models/
│ └── forgery_detector_model.h5
│
├── src/
│ ├── preprocessing.py # Laplacian preprocessing
│ ├── model.py # CNN model architecture
│ ├── train.py # Training pipeline
│ └── gradcam.py # Prediction + Grad-CAM visualization
│
├── main.py
├── .gitignore
└── README.md

---

## 🧠 Model Workflow

1. **Input Image**
2. Laplacian preprocessing (edge/artifact emphasis)
3. CNN classification (Authentic / Forged)
4. Confidence & decision margin computation
5. Grad-CAM heatmap generation for explanation

---

## 📊 Output Metrics Explained

- **Prediction Score**  
  Raw model output (0 → Authentic, 1 → Forged)

- **Confidence (%)**  
  How strongly the model supports its prediction

- **Decision Margin**  
  Distance from the decision boundary (0.5)

- **Heatmap Intensity Summary**
  - Mean intensity → overall attention strength
  - Max intensity → strongest suspicious region

---

## 🖼️ Grad-CAM Visualization

- 🔵 Low attention → less influence
- 🔴 High attention → regions influencing the decision
- Helps explain *why* an image is classified as forged or authentic

---

## 🚀 How to Run

### 1️⃣ Train the model
```bash
python src/train.py

2️⃣ Run forgery detection + Grad-CAM
python src/gradcam.py

📱 Deployment Plan (Final Stage)

Model runs on a FastAPI backend

Users upload images via a web interface

Works on mobile phones without an app

No TensorFlow Lite required

📦 Dataset

CASIA Image Tampering Dataset

Includes copy-move and splicing forgeries

Custom forged images also supported

👨‍💻 Author

Built as an academic project to demonstrate:

Practical deep learning

Explainable AI (XAI)

Real-world deployment readiness
📝 License

This project is for educational and academic use.


---

## ✅ What to do now

In PowerShell (inside your project folder):

```powershell
notepad README.md
