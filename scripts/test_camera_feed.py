import cv2
import os

os.makedirs("logs",exist_ok=True)
cap=cv2.VideoCapture(0) #If using an IP camera, use cv2.VideoCapture("rtsp://username:pass@ip:554/stream").

if not cap.isOpened():
    print("ERROR: Cannot open camera. Try changing device index (0/1) or use RTSP URL.")
    exit(1)
ret, frame = cap.read()
if not ret:
    print("ERROR: Can't read a frame from camera.")
else:
    print("Captured frame shape:", frame.shape)
    out_path = "logs/sample_frame.jpg"
    cv2.imwrite(out_path, frame)
    print("Saved test frame to", out_path)

cap.release()