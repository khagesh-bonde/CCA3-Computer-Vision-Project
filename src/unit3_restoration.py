from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "data" / "images" / "my_photo.jpeg"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "unit3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

original = cv2.imread(str(IMAGE_PATH))

if original is None:
    raise FileNotFoundError(f"Photo not found: {IMAGE_PATH}")

height, width = original.shape[:2]
if max(height, width) > 1000:
    scale = 1000 / max(height, width)
    original = cv2.resize(original, (int(width * scale), int(height * scale)))

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
rng = np.random.default_rng(42)

# 1. Gaussian noise
noise = rng.normal(0, 20, gray.shape)
gaussian_noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
gaussian_restored = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)

# 2. Salt-and-pepper noise
sp_noisy = gray.copy()
random_values = rng.random(gray.shape)
sp_noisy[random_values < 0.02] = 0
sp_noisy[random_values > 0.98] = 255
sp_restored = cv2.medianBlur(sp_noisy, 5)

# 3. PSNR and SSIM comparison
def evaluate(reference, restored):
    psnr = peak_signal_noise_ratio(reference, restored, data_range=255)
    ssim = structural_similarity(reference, restored, data_range=255)
    return psnr, ssim

for name, noisy, restored in [
    ("Gaussian", gaussian_noisy, gaussian_restored),
    ("Salt-and-Pepper", sp_noisy, sp_restored)
]:
    psnr, ssim = evaluate(gray, restored)
    print(f"{name} restoration: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

# 4. Thresholding
_, binary = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 5. Morphological operations
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 6. Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# 7. Watershed segmentation
color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
_, sure_bg = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(
    dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY
)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(color_image, markers)

watershed_result = color_image.copy()
watershed_result[markers == -1] = [0, 0, 255]

# 8. Save outputs
results = {
    "original_gray.png": gray,
    "gaussian_noisy.png": gaussian_noisy,
    "gaussian_restored.png": gaussian_restored,
    "salt_pepper_noisy.png": sp_noisy,
    "median_restored.png": sp_restored,
    "threshold.png": binary,
    "erosion.png": erosion,
    "dilation.png": dilation,
    "opening.png": opening,
    "closing.png": closing,
    "canny_edges.png": edges,
    "watershed.png": watershed_result
}

for filename, image in results.items():
    cv2.imwrite(str(OUTPUT_DIR / filename), image)

# 9. Display comparison
images = [
    (gray, "Original"),
    (gaussian_noisy, "Gaussian Noise"),
    (gaussian_restored, "Gaussian Filter"),
    (sp_noisy, "Salt-Pepper Noise"),
    (sp_restored, "Median Filter"),
    (binary, "Threshold"),
    (erosion, "Erosion"),
    (dilation, "Dilation"),
    (opening, "Opening"),
    (closing, "Closing"),
    (edges, "Canny Edges"),
    (watershed_result, "Watershed")
]

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for ax, (image, title) in zip(axes.flat, images):
    if len(image.shape) == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "unit3_comparison.png", dpi=120)
plt.show()

print(f"\nAll Unit 3 outputs saved to: {OUTPUT_DIR}")