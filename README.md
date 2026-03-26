# Automated Zuma Player

A real-time computer vision system that automates Zuma gameplay using classical computer vision techniques, without relying on machine learning.

## Overview
This project was designed to build an autonomous gameplay pipeline for Zuma by analyzing the game screen in real time, detecting relevant objects, understanding motion, and making shooting decisions automatically.

The system focuses on modular computer vision components that work together to capture the game state, identify targets, and respond under dynamic conditions.

## Key Features
- Real-time screen capture and game-state analysis
- Robust ROI detection and tracking
- Motion-based target localization
- Ball detection using Hough Circles
- HSV-based color classification with confidence scoring
- Chain analysis for target selection
- Automated shooting decision logic
- Path extraction experiments using image processing techniques

## Methods Used
- Contour filtering
- Temporal smoothing
- Motion accumulation
- Dense Optical Flow
- Hough Circle detection
- HSV color analysis
- Morphological operations
- Thresholding
- Distance transform
- Skeletonization
- Graph-based centerline extraction

## Tech Stack
- Python
- OpenCV
- NumPy

## Project Structure
- `roi.py` – ROI detection and tracking
- `detectShooter.py` – shooter position / motion-based localization
- `ball_detector.py` – ball detection and color analysis
- `roinly.py` – ROI-focused processing
- main integration script – combines modules into a real-time pipeline

## What I Learned
This project strengthened my experience in:
- real-time image processing
- modular computer vision system design
- algorithmic problem solving
- motion analysis and object detection
- decision-making under dynamic visual conditions

## Screenshots
![ROI Detection](assets/roi-detection.png)
![Ball Detection](assets/ball-detected.png)
![Shooter Detection](assets/shooter-detected.png)


## Notes
This project uses classical computer vision only and does not rely on machine learning models.
