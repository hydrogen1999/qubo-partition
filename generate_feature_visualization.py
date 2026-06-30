from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import os

img_path = "datasets/casia_v2/CASIA2/Tp/Tp_D_CNN_M_N_art00052_arc00030_11853.jpg"

os.makedirs("feature_visualization", exist_ok=True)

# ---------- Original ----------
img = Image.open(img_path).convert("RGB")
img.save("feature_visualization/original.png")

# ---------- ELA ----------
temp_path = "feature_visualization/temp.jpg"
img.save(temp_path, "JPEG", quality=90)

recompressed = Image.open(temp_path)

ela = ImageChops.difference(img, recompressed)

extrema = ela.getextrema()
max_diff = max([ex[1] for ex in extrema])

if max_diff == 0:
    max_diff = 1

scale = 255.0 / max_diff
ela = ImageEnhance.Brightness(ela).enhance(scale)

ela.save("feature_visualization/ela.png")

# ---------- Noise Residual ----------
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

noise = cv2.absdiff(gray, blur)

noise = cv2.normalize(
    noise,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

cv2.imwrite(
    "feature_visualization/noise.png",
    noise
)

print("Saved:")
print("feature_visualization/original.png")
print("feature_visualization/ela.png")
print("feature_visualization/noise.png")