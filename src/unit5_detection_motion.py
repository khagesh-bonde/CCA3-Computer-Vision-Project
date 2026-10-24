
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "data" / "images" / "my_photo.jpeg"
VIDEO = ROOT / "data" / "videos" / "my_video.mp4"
OUT = ROOT / "outputs" / "unit5"
OUT.mkdir(parents=True, exist_ok=True)

# Check input files
if not IMAGE.exists():
    raise FileNotFoundError(f"Photo not found: {IMAGE}")
if not VIDEO.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO}")

# 1. Haar Cascade face detection
img = cv2.imread(str(IMAGE))
if img is None:
    raise ValueError("Cannot read photo.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
faces = cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5
)

haar = img.copy()
for x, y, w, h in faces:
    cv2.rectangle(haar, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(haar, "Face", (x, max(y-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite(str(OUT / "haar_face_detection.jpg"), haar)
print("Haar Cascade faces:", len(faces))

# 2. YOLO object detection
print("Loading YOLO model; first run may download it...")
model = YOLO("yolo11n.pt")

result = model.predict(source=str(IMAGE), conf=0.25, verbose=False)[0]
cv2.imwrite(str(OUT / "yolo_photo_detection.jpg"), result.plot())

with open(OUT / "yolo_detection_results.txt", "w", encoding="utf-8") as f:
    for box in result.boxes:
        label = model.names[int(box.cls[0].item())]
        confidence = float(box.conf[0].item())
        line = f"{label}: confidence={confidence:.3f}"
        print(line)
        f.write(line + "\n")

# 3, 4, 5. Video motion analysis
cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError("Cannot open my_video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

ok, first = cap.read()
if not ok:
    raise RuntimeError("Video has no readable frames.")

if first.shape[1] > 960:
    first = cv2.resize(
        first, (960, int(first.shape[0] * 960 / first.shape[1]))
    )

previous_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
first_gray = previous_gray.copy()

mog = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=25, detectShadows=True
)
mog.apply(first)

points = cv2.goodFeaturesToTrack(
    previous_gray, maxCorners=100, qualityLevel=0.3,
    minDistance=7, blockSize=7
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out = cv2.VideoWriter(
    str(OUT / "motion_analysis.mp4"), fourcc, fps,
    (first.shape[1], first.shape[0])
)

frame_diff_saved = False
frame_count = 1
kernel = np.ones((3, 3), np.uint8)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame.shape[1] > 960:
        frame = cv2.resize(
            frame, (960, int(frame.shape[0] * 960 / frame.shape[1]))
        )

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame differencing
    diff = cv2.absdiff(previous_gray, current_gray)
    _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # GMM background subtraction
    foreground = mog.apply(frame)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, kernel
    )

    # Lucas-Kanade optical flow
    flow_frame = frame.copy()
    if points is not None and len(points) > 0:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if new_points is not None and status is not None:
            good_new = new_points[status.ravel() == 1]
            good_old = points[status.ravel() == 1]

            for new, old in zip(good_new, good_old):
                x1, y1 = old.ravel()
                x2, y2 = new.ravel()
                cv2.arrowedLine(
                    flow_frame, (int(x1), int(y1)),
                    (int(x2), int(y2)), (0, 255, 0), 2
                )

            points = good_new.reshape(-1, 1, 2)

    if points is None or len(points) < 10:
        points = cv2.goodFeaturesToTrack(
            current_gray, maxCorners=100, qualityLevel=0.3,
            minDistance=7, blockSize=7
        )

    if not frame_diff_saved:
        cv2.imwrite(str(OUT / "frame_differencing.png"), diff_mask)
        cv2.imwrite(str(OUT / "gmm_background_subtraction.png"), foreground)
        cv2.imwrite(str(OUT / "lucas_kanade_optical_flow.png"), flow_frame)
        cv2.imwrite(str(OUT / "motion_sample_frame.jpg"), frame)
        frame_diff_saved = True

    cv2.putText(
        flow_frame, "Lucas-Kanade Optical Flow", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    video_out.write(flow_frame)

    previous_gray = current_gray.copy()
    frame_count += 1

cap.release()
video_out.release()

# YOLO on a frame from the video
sample = OUT / "motion_sample_frame.jpg"
video_result = model.predict(
    source=str(sample), conf=0.25, verbose=False
)[0]
cv2.imwrite(str(OUT / "yolo_video_detection.jpg"), video_result.plot())

print("Frames processed:", frame_count)
print("UNIT 5 COMPLETED!")
print("All results saved to:", OUT)
