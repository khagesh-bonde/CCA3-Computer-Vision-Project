
from pathlib import Path
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE = ROOT / "data" / "images" / "my_photo.jpeg"
VIDEO = ROOT / "data" / "videos" / "my_video.mp4"

OUT = ROOT / "outputs" / "unit5"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("UNIT 5: OBJECT DETECTION AND MOTION ANALYSIS")
print("=" * 60)
print("Project folder:", ROOT)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not IMAGE.exists():
    raise FileNotFoundError(
        f"Photo not found: {IMAGE}\n"
        "Please check data/images/my_photo.jpeg"
    )

if not VIDEO.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO}\n"
        "Please check data/videos/my_video.mp4"
    )


# ============================================================
# 1. HAAR CASCADE FACE DETECTION
# ============================================================

print("\n[1/4] Running Haar Cascade face detection...")

img = cv2.imread(str(IMAGE))

if img is None:
    raise ValueError(f"Cannot read photo: {IMAGE}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Locate the Haar Cascade XML file.
cascade_filename = "haarcascade_frontalface_default.xml"
cascade_path = Path(cv2.data.haarcascades) / cascade_filename

# If the XML file is missing, download the official OpenCV file.
if not cascade_path.exists():
    print("Haar Cascade XML not found. Downloading it...")

    cascade_path = OUT / cascade_filename
    cascade_url = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "4.x/data/haarcascades/"
        "haarcascade_frontalface_default.xml"
    )

    try:
        urllib.request.urlretrieve(cascade_url, str(cascade_path))
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the Haar Cascade XML file. "
            "Check your internet connection and try again."
        ) from exc

cascade = cv2.CascadeClassifier(str(cascade_path))

if cascade.empty():
    raise RuntimeError(
        f"Could not load Haar Cascade XML: {cascade_path}"
    )

faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

haar_output = img.copy()

for x, y, w, h in faces:
    cv2.rectangle(
        haar_output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        haar_output,
        "Face",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

haar_path = OUT / "haar_face_detection.jpg"

if not cv2.imwrite(str(haar_path), haar_output):
    raise RuntimeError(f"Could not save: {haar_path}")

print("Faces detected:", len(faces))
print("Saved:", haar_path)


# ============================================================
# 2. YOLO OBJECT DETECTION ON PHOTO
# ============================================================

print("\n[2/4] Running YOLO object detection on photo...")
print("Loading YOLO model; first run may download model weights.")

model_path = ROOT / "yolo11n.pt"

if not model_path.exists():
    # Ultralytics can download the official pretrained weights.
    model_path = "yolo11n.pt"

model = YOLO(str(model_path))

photo_result = model.predict(
    source=str(IMAGE),
    conf=0.25,
    verbose=False
)[0]

yolo_photo_path = OUT / "yolo_photo_detection.jpg"

if not cv2.imwrite(str(yolo_photo_path), photo_result.plot()):
    raise RuntimeError(f"Could not save: {yolo_photo_path}")

photo_results_path = OUT / "yolo_detection_results.txt"

with open(photo_results_path, "w", encoding="utf-8") as file:
    if photo_result.boxes is None or len(photo_result.boxes) == 0:
        print("No objects detected in the photo.")
        file.write("No objects detected.\n")

    else:
        for box in photo_result.boxes:
            class_id = int(box.cls[0].item())
            label = model.names[class_id]
            confidence = float(box.conf[0].item())

            line = f"{label}: confidence={confidence:.3f}"
            print(line)
            file.write(line + "\n")

print("Saved:", yolo_photo_path)
print("Saved:", photo_results_path)


# ============================================================
# 3. VIDEO MOTION ANALYSIS
#    A. FRAME DIFFERENCING
#    B. GMM BACKGROUND SUBTRACTION
#    C. LUCAS-KANADE OPTICAL FLOW
# ============================================================

print("\n[3/4] Running video motion analysis...")

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

if not np.isfinite(fps) or fps <= 0:
    fps = 25.0

ok, first_frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError("Video has no readable frames.")

# Resize frames consistently for manageable processing.
def resize_frame(frame, max_width=960):
    height, width = frame.shape[:2]

    if width > max_width:
        new_height = int(height * max_width / width)
        frame = cv2.resize(
            frame,
            (max_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    return frame


first_frame = resize_frame(first_frame)
frame_height, frame_width = first_frame.shape[:2]

previous_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)

# GMM background subtractor.
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=True
)

# Initialize background model.
background_subtractor.apply(first_frame)

# Initialize Lucas-Kanade tracking points.
points = cv2.goodFeaturesToTrack(
    previous_gray,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Output video.
motion_video_path = OUT / "motion_analysis.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(motion_video_path),
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not video_writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not create motion_analysis.mp4. "
        "Check video codec support and output folder permissions."
    )

# Output images.
frame_diff_path = OUT / "frame_differencing.png"
gmm_path = OUT / "gmm_background_subtraction.png"
optical_flow_path = OUT / "lucas_kanade_optical_flow.png"
sample_frame_path = OUT / "motion_sample_frame.jpg"

kernel = np.ones((3, 3), dtype=np.uint8)

frame_count = 1
saved_sample = False

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame = resize_frame(frame)

    # Keep the output frame size consistent.
    if frame.shape[:2] != (frame_height, frame_width):
        frame = cv2.resize(
            frame,
            (frame_width, frame_height)
        )

    current_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # A. FRAME DIFFERENCING
    # --------------------------------------------------------

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    _, difference_mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small isolated noise.
    difference_mask = cv2.morphologyEx(
        difference_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # B. GMM BACKGROUND SUBTRACTION
    # --------------------------------------------------------

    foreground_mask = background_subtractor.apply(frame)

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # C. LUCAS-KANADE OPTICAL FLOW
    # --------------------------------------------------------

    flow_frame = frame.copy()

    if points is not None and len(points) > 0:
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        if (
            new_points is not None
            and status is not None
        ):
            valid = status.ravel() == 1

            good_new = new_points[valid]
            good_old = points[valid]

            for new_point, old_point in zip(good_new, good_old):
                x1, y1 = old_point.ravel()
                x2, y2 = new_point.ravel()

                cv2.arrowedLine(
                    flow_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

            points = good_new.reshape(-1, 1, 2)

        else:
            points = None

    # Re-detect tracking points when too few remain.
    if points is None or len(points) < 10:
        points = cv2.goodFeaturesToTrack(
            current_gray,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

    # Add a label to the optical flow output.
    cv2.putText(
        flow_frame,
        "Lucas-Kanade Optical Flow",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Save outputs from the first processed pair of frames.
    if not saved_sample:
        cv2.imwrite(str(frame_diff_path), difference_mask)
        cv2.imwrite(str(gmm_path), foreground_mask)
        cv2.imwrite(str(optical_flow_path), flow_frame)
        cv2.imwrite(str(sample_frame_path), frame.copy())

        saved_sample = True

    # Write each annotated frame to the output video.
    video_writer.write(flow_frame)

    previous_gray = current_gray.copy()
    frame_count += 1

cap.release()
video_writer.release()

print("Frames processed:", frame_count)
print("Saved:", frame_diff_path)
print("Saved:", gmm_path)
print("Saved:", optical_flow_path)
print("Saved:", motion_video_path)


# ============================================================
# 4. YOLO OBJECT DETECTION ON A VIDEO FRAME
# ============================================================

print("\n[4/4] Running YOLO detection on a video frame...")

if not sample_frame_path.exists():
    raise RuntimeError("Motion sample frame was not generated.")

video_result = model.predict(
    source=str(sample_frame_path),
    conf=0.25,
    verbose=False
)[0]

yolo_video_path = OUT / "yolo_video_detection.jpg"

if not cv2.imwrite(str(yolo_video_path), video_result.plot()):
    raise RuntimeError(f"Could not save: {yolo_video_path}")

print("Saved:", yolo_video_path)


# ============================================================
# COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("UNIT 5 PROCESSING COMPLETED")
print("=" * 60)
print("Output folder:", OUT)
print("Check the generated images, text file and MP4 video.")
