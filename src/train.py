import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing import apply_laplacian
from model import build_model

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ✅ NEW: model directory + path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "forgery_detector_model.h5")

IMAGE_SIZE = (224, 224)
EPOCHS = 3
BATCH_SIZE = 16


def load_dataset():
    images = []
    labels = []

    for label, folder in enumerate(["authentic", "forged"]):
        folder_path = os.path.join(DATASET_DIR, folder)

        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, file)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.resize(img, IMAGE_SIZE)

                # Laplacian preprocessing
                img = apply_laplacian(img)

                img = img / 255.0
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


def main():
    print("[INFO] Loading dataset...")
    X, y = load_dataset()

    if len(X) == 0:
        raise ValueError("Dataset is empty. Add images to dataset/authentic and dataset/forged.")

    print(f"[INFO] Total samples: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Building model...")
    model = build_model(input_shape=(224, 224, 3))

    print("[INFO] Training started...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    print("[INFO] Training completed.")

    # ✅ Ensure models folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ✅ Save model in models/
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
