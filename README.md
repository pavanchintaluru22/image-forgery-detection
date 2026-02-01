
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

## 🚀 How to Run

### 1️⃣ Train the model
```bash
python src/train.py
````

### 2️⃣ Run forgery detection + Grad-CAM

```bash
python src/gradcam.py
```

---

## 📱 Deployment Plan (Final Stage)

* Model runs on a **FastAPI backend**
* Users upload images via a web interface
* Works on **mobile phones without an app**
* **No TensorFlow Lite required**

---

## 📦 Dataset

* **CASIA Image Tampering Dataset**
* Includes **copy-move** and **splicing** forgeries
* Custom forged images also supported

---

## 👨‍💻 Author

Built as an academic project to demonstrate:

* Practical deep learning
* Explainable AI (XAI)
* Real-world deployment readiness

---

## 📝 License

This project is for **educational and academic use**.

````



