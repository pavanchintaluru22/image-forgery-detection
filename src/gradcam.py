import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import apply_laplacian

# =======================
# PATHS
# =======================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "forgery_detector_model.h5"
)


# 🔴 CHANGE THIS IMAGE WHEN NEEDED
IMAGE_PATH = r"C:\Users\pchin\PycharmProjects\PythonProject1\dataset\forged\Tp_D_CRD_S_O_ani10111_ani10103_10635.jpg"

IMAGE_SIZE = (224, 224)
DISPLAY_SIZE = (500, 500)

# =======================
# LOAD MODEL
# =======================
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# =======================
# LOAD & PREPROCESS IMAGE
# =======================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("❌ Image not found. Check IMAGE_PATH.")

original_img = cv2.resize(img, IMAGE_SIZE)

# SAME preprocessing as training
img_proc = apply_laplacian(original_img)
img_proc = img_proc / 255.0
img_proc = np.expand_dims(img_proc, axis=0)

# =======================
# PREDICTION
# =======================
pred = model.predict(img_proc)[0][0]

if pred >= 0.5:
    predicted_class = "FORGED"
    confidence = pred
else:
    predicted_class = "AUTHENTIC"
    confidence = 1 - pred

decision_margin = abs(pred - 0.5)

print("\n[RESULT]")
print(f"Predicted class      : {predicted_class}")
print(f"Prediction score     : {pred:.4f}")
print(f"Confidence (%)       : {confidence * 100:.2f}%")
print(f"Decision margin      : {decision_margin:.3f}")

# =======================
# GRAD-CAM
# =======================
# Find last convolution layer safely
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

if last_conv_layer is None:
    raise RuntimeError("❌ No Conv2D layer found for Grad-CAM")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_proc)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

mean_intensity = np.mean(heatmap)
max_intensity = np.max(heatmap)

print("\n[HEATMAP INTENSITY SUMMARY]")
print(f"Mean heatmap intensity : {mean_intensity:.4f}")
print(f"Max heatmap intensity  : {max_intensity:.4f}")
print("\n[PLAIN-ENGLISH EXPLANATION]")
print("-" * 40)

# 1. Prediction score explanation
print(
    f"• Prediction score ({pred:.2f}) represents the raw probability "
    f"given by the model that this image is FORGED."
)

# 2. Confidence explanation
print(
    f"• Confidence ({confidence * 100:.1f}%) shows how sure the model is "
    f"about its final decision."
)

# 3. Decision margin explanation
if decision_margin < 0.1:
    margin_text = "very close to the decision boundary (low certainty)"
elif decision_margin < 0.25:
    margin_text = "moderately away from the boundary (medium certainty)"
else:
    margin_text = "far from the boundary (high certainty)"

print(
    f"• Decision margin ({decision_margin:.3f}) tells how strongly the "
    f"model favors one class over the other — here it is {margin_text}."
)

# 4. Heatmap intensity explanation
print(
    f"• Mean heatmap intensity ({mean_intensity:.3f}) indicates how much of "
    f"the image the model relied on overall while making the decision."
)

print(
    f"• Max heatmap intensity ({max_intensity:.3f}) shows the strongest "
    f"single region that influenced the prediction."
)

# 5. Final human-readable summary
print("\n[FINAL INTERPRETATION]")
if confidence > 0.7 and mean_intensity > 0.25:
    print(
        "The model is confident and focuses strongly on specific regions, "
        "which suggests visible manipulation patterns."
    )
elif confidence > 0.6:
    print(
        "The model moderately supports its decision, but the evidence is "
        "not extremely strong."
    )
else:
    print(
        "The model is uncertain, and the detected cues are weak. "
        "This image may require manual verification."
    )


# =======================
# CREATE OVERLAY
# =======================
heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    original_img, 0.6,
    heatmap_colored, 0.4,
    0
)

# =======================
# DISPLAY (FIXED SIZE + CLEAN TEXT)
# =======================
orig_disp = cv2.resize(original_img, DISPLAY_SIZE)
heat_disp = cv2.resize(overlay, DISPLAY_SIZE)

label = f"{predicted_class} | Confidence: {confidence * 100:.1f}%"

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 1
padding = 8

(text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

# Background for text
cv2.rectangle(
    heat_disp,
    (0, 0),
    (text_w + padding * 2, text_h + padding * 2),
    (0, 0, 0),
    -1
)

# Text
cv2.putText(
    heat_disp,
    label,
    (padding, text_h + padding),
    font,
    font_scale,
    (255, 255, 255),
    thickness,
    cv2.LINE_AA
)

cv2.imshow("Original Image", orig_disp)
cv2.imshow("Grad-CAM Explanation", heat_disp)

print("\n[INFO] Press any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()
