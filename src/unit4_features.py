
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE1_PATH = PROJECT_DIR / "data" / "images" / "my_photo.jpeg"
IMAGE2_PATH = PROJECT_DIR / "data" / "images" / "second_photo.jpeg"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "unit4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def resize_for_processing(image, max_size=800):
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    return image


def get_sift_features(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(img_a, kp_a, des_a, img_b, kp_b, des_b, name):
    if des_a is None or des_b is None:
        print(f"{name}: descriptors unavailable")
        return 0, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des_a, des_b, k=2)

    good_matches = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    visual = cv2.drawMatches(
        img_a, kp_a,
        img_b, kp_b,
        good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    output_path = OUTPUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(output_path), visual)

    print(f"{name}: {len(good_matches)} good matches")
    print(f"Saved: {output_path}")

    return len(good_matches), visual


# Load and resize original photographs
img1 = resize_for_processing(load_image(IMAGE1_PATH))
img2 = resize_for_processing(load_image(IMAGE2_PATH))

sift = cv2.SIFT_create()

# HOG features
hog = cv2.HOGDescriptor()
hog1 = hog.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
hog2 = hog.compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

print("=" * 50)
print("HOG FEATURES")
print("Image 1 HOG shape:", hog1.shape)
print("Image 2 HOG shape:", hog2.shape)

# Original SIFT keypoints and descriptors
kp1, des1 = get_sift_features(img1, sift)
kp2, des2 = get_sift_features(img2, sift)

print("\nSIFT FEATURES")
print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

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

# 1. Original matching
original_count, original_visual = match_features(
    img1, kp1, des1,
    img2, kp2, des2,
    "matching_original"
)

# 2. Rotate image 1 by 90 degrees and match against image 2
rotated = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(str(OUTPUT_DIR / "rotated_image.jpg"), rotated)

kp_rot, des_rot = get_sift_features(rotated, sift)

rotated_count, rotated_visual = match_features(
    rotated, kp_rot, des_rot,
    img2, kp2, des2,
    "matching_rotated"
)

# 3. Resize image 1 to 75% and match against image 2
resized = cv2.resize(
    img1, None,
    fx=0.75, fy=0.75,
    interpolation=cv2.INTER_AREA
)
cv2.imwrite(str(OUTPUT_DIR / "resized_image.jpg"), resized)

kp_res, des_res = get_sift_features(resized, sift)

resized_count, resized_visual = match_features(
    resized, kp_res, des_res,
    img2, kp2, des2,
    "matching_resized"
)

# Save comparison table as CSV
import csv

comparison_rows = [
    ["Test", "Image 1 keypoints", "Image 2 keypoints", "Good matches"],
    ["Original", len(kp1), len(kp2), original_count],
    ["Rotated 90 degrees", len(kp_rot), len(kp2), rotated_count],
    ["Resized to 75%", len(kp_res), len(kp2), resized_count],
]

csv_path = OUTPUT_DIR / "sift_matching_comparison.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(comparison_rows)

print("\nSIFT MATCHING COMPARISON")
for row in comparison_rows:
    print(" | ".join(map(str, row)))

# Save a combined visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].imshow(cv2.cvtColor(keypoints1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Image 1: SIFT Keypoints")

axes[0, 1].imshow(cv2.cvtColor(keypoints2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Image 2: SIFT Keypoints")

if original_visual is not None:
    axes[1, 0].imshow(
        cv2.cvtColor(original_visual, cv2.COLOR_BGR2RGB)
    )
    axes[1, 0].set_title(f"Original Matching: {original_count}")

if rotated_visual is not None:
    axes[1, 1].imshow(
        cv2.cvtColor(rotated_visual, cv2.COLOR_BGR2RGB)
    )
    axes[1, 1].set_title(f"Rotated Matching: {rotated_count}")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "unit4_comparison.png", dpi=150)
plt.close()

print("\nUnit 4 outputs saved to:", OUTPUT_DIR)
print("CSV comparison saved to:", csv_path)
