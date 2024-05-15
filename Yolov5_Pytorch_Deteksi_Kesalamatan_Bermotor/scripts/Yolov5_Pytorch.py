import torch
import cv2
import numpy as np
from mss import mss
import pathlib
import os
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load(r'C:\Users\OmenX2s\Desktop\ai\scripts\yolov5', 'custom', path=r'C:\Users\OmenX2s\Desktop\ai\scripts\best.pt', source='local')

# Define class names
class_names = ['Helm', 'Kendaraan', 'Pakai Sepatu', 'Pakaian Terbuka', 'Pakaian Tertutup', 'Spion', 'Tidak Pakai Helm', 'Tidak Pakai Sepatu']

# Choose source: webcam or video file
source_type = input("Enter 'webcam' to use webcam or 'mp4' to use a video file: ")

if source_type == 'webcam':
    cap = cv2.VideoCapture(0)  # Access webcam
elif source_type == 'mp4':
    # Open video file
    video_path = r'C:\Users\OmenX2s\Desktop\ai\scripts\video.mp4'
    cap = cv2.VideoCapture(video_path)
else:
    print("Invalid source type. Please choose 'webcam' or 'mp4'.")
    exit()

# Create directory for screenshots if it doesn't exist
screenshot_dir = r'C:\Users\OmenX2s\Desktop\ai\ScreenShoot'
os.makedirs(screenshot_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, size=640)
    df = results.pandas().xyxy[0]
    
    for i in range(len(df)):
        xmin = int(df.iloc[i,0])
        ymin = int(df.iloc[i,1])
        xmax = int(df.iloc[i,2])
        ymax = int(df.iloc[i,3])
        class_idx = int(df.iloc[i,5])
        class_name = class_names[class_idx]
        
        # Set color based on class name
        if class_name in ['Pakaian Terbuka', 'Tidak Pakai Helm', 'Tidak Pakai Sepatu']:
            color = (0, 0, 255)  # Red color for specified classes
        else:
            color = (255, 0, 0)  # Blue color for other classes
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save screenshot if specific classes are detected
        if class_name in ['Pakaian Terbuka', 'Tidak Pakai Helm', 'Tidak Pakai Sepatu']:
            screenshot_path = os.path.join(screenshot_dir, f'screenshot_{class_name}_{i}.png')
            cv2.imwrite(screenshot_path, frame)


    # Display the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
