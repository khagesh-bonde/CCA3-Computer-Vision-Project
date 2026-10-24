# Computer Vision Project – MCA CCA3

## Project Overview
This project demonstrates image processing, feature extraction, object detection, and motion analysis using Python and computer vision libraries.

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-image
- Ultralytics YOLO

## Unit 3: Image Processing
- Image noise generation and restoration
- Gaussian and median filtering
- PSNR and SSIM evaluation
- Image thresholding
- Morphological operations: erosion, dilation, opening, and closing
- Canny edge detection
- Watershed segmentation

## Unit 4: Feature Extraction
- Histogram of Oriented Gradients (HOG)
- Scale-Invariant Feature Transform (SIFT)
- Feature matching between images
- Rotation and resizing experiments

## Unit 5: Object Detection and Motion Analysis
- Face detection using Haar Cascade
- Object detection using YOLO
- Frame differencing
- Gaussian Mixture Model (GMM) background subtraction
- Lucas–Kanade optical flow

## Project Structure
```text
CCA3_CV_Project/
├── data/
│   ├── images/
│   └── videos/
├── outputs/
│   ├── unit3/
│   ├── unit4/
│   └── unit5/
├── src/
├── requirements.txt
└── README.md
```

## How to Run
1. Install Python 3.11.
2. Create and activate a virtual environment.
3. Install dependencies using:

   `python -m pip install -r requirements.txt`

4. Run the required unit scripts from the project root:

   `python src/unit3_restoration.py`

   `python src/unit4_features.py`

   `python src/unit5_detection_motion.py`

## Outputs
Generated images, comparisons, detection results, and motion analysis videos are saved in the corresponding folders inside `outputs`.

## Author
MCA Student – Computer Applications
