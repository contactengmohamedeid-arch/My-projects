# Virtual Security Camera
Real-time person detection on a webcam with YOLOv8n (COCO class 0, confidence 0.5). Detected people get a red box labelled "Intruder!" and a frame is saved to an `Intruders/` folder, with a 5-second cooldown between saves. Press Q to quit.

**Run:** `pip install opencv-python ultralytics`, open `Virtual Security Camera.ipynb`, run all cells. Uses `yolov8n.pt`.
