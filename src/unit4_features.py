from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE1_PATH = PROJECT_DIR / "data" / "images" / "my_photo.jpeg"
IMAGE2_PATH = PROJECT_DIR / "data" / "images" / "second_photo.jpeg"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "unit4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

img1 = cv2.imread(str(IMAGE1_PATH))
img2 = cv2.imread(str(IMAGE2_PATH))

if img1 is None:
    raise FileNotFoundError(f"First image not found: {IMAGE1_PATH}")

if img2 is None:
    raise FileNotFoundError(f"Second image not found: {IMAGE2_PATH}")

# Resize images to save memory
def resize_image(image):
    h, w = image.shape[:2]
    scale = min(1, 800 / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

img1 = resize_image(img1)
img2 = resize_image(img2)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 1. HOG descriptor
hog = cv2.HOGDescriptor()
hog1 = hog.compute(gray1)
hog2 = hog.compute(gray2)

print("HOG FEATURES")
print("Image 1 HOG shape:", hog1.shape)
print("Image 2 HOG shape:", hog2.shape)

# 2. SIFT keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("\nSIFT FEATURES")
print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

# Draw keypoints
keypoints1 = cv2.drawKeypoints(
    img1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
keypoints2 = cv2.drawKeypoints(
    img2, kp2, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(str(OUTPUT_DIR / "image1_keypoints.jpg"), keypoints1)
cv2.imwrite(str(OUTPUT_DIR / "image2_keypoints.jpg"), keypoints2)

# 3. Match SIFT descriptors
if des1 is not None and des2 is not None:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    matched_image = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "feature_matching.jpg"),
        matched_image
    )
    print("Good matches:", len(good_matches))

else:
    print("Not enough features found to match the images.")

# 4. Rotate and resize image 1, then compare SIFT matching
rotated = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
resized = cv2.resize(
    img1,
    None,
    fx=0.75,
    fy=0.75,
    interpolation=cv2.INTER_AREA
)

cv2.imwrite(str(OUTPUT_DIR / "rotated_image.jpg"), rotated)
cv2.imwrite(str(OUTPUT_DIR / "resized_image.jpg"), resized)

print("\nRotation and resized test images saved.")
print("All Unit 4 outputs saved to:", OUTPUT_DIR)

# 5. Display keypoint and matching results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(keypoints1, cv2.COLOR_BGR2RGB))
axes[0].set_title("Image 1 - SIFT Keypoints")

axes[1].imshow(cv2.cvtColor(keypoints2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Image 2 - SIFT Keypoints")

if des1 is not None and des2 is not None:
    axes[2].imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("SIFT Feature Matching")
else:
    axes[2].text(0.1, 0.5, "No descriptors available")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "unit4_comparison.png", dpi=120)
plt.show()
