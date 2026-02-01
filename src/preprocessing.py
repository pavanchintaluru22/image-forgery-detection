import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def load_image(path, size=(224, 224)):
    """
    Loads an image from disk and resizes it.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to load image at {path}")
    image = cv2.resize(image, size)
    return image


def apply_laplacian(image):
    """
    Applies Laplacian filter to highlight high-frequency artifacts.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Normalize to 0–255
    laplacian = np.absolute(laplacian)
    laplacian = (laplacian / laplacian.max()) * 255
    laplacian = laplacian.astype(np.uint8)

    # Convert back to 3 channels (CNN expects RGB-like input)
    laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    return laplacian_3ch


def visualize(original, laplacian):
    """
    Shows original vs Laplacian image.
    """
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Laplacian (Artifacts Highlighted)")
    plt.imshow(cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick test with a sample image (put one image path here)
    test_image_path = r"C:\Users\pchin\PyCharmProjects\PythonProject1\dataset\authentic\sample.jpg"

    img = load_image(test_image_path)
    lap = apply_laplacian(img)
    visualize(img, lap)
