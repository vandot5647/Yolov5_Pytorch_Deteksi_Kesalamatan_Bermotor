import torch
import cv2
import os
import pathlib
import time
import seaborn as sns

# Temporary workaround for pathlib on Windows
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model, path=r'D:\danu\_internal\best.pt', source='local')
model = torch.hub.load('_internal/yolov5', 'custom', path='_internal/best.pt', source='local')

# Define class names
class_names = [
    'Helm', 'Kendaraan', 'Pakai Sepatu', 'Pakaian Terbuka',
    'Pakaian Tertutup', 'Spion', 'Tidak Pakai Helm', 'Tidak Pakai Sepatu'
]

print("Deteksi Keselamatan Bermotor.")

# Choose source: webcam or video file
source_type = input("Enter 'webcam' to use webcam or 'mp4' to use a video file: ")

if source_type == 'webcam':
    cap = cv2.VideoCapture(0)  # Access webcam
elif source_type == 'mp4':
    # Get video file path from user
    video_path = input("Enter the path to the video file (Example D:\\danu\\video.mp4) : ")
    # Check if the file extension is a video format
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Invalid file format. Please enter a path to a video file with a supported extension (.mp4, .avi, .mov, .mkv).")
        exit()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file at {video_path}. Please check the file path and ensure it is correct.")
        exit()
else:
    print("Invalid source type. Please choose 'webcam' or 'mp4'.")
    exit()

# Get directory for saving screenshots from user
screenshot_dir = input(r"Enter the directory to save screenshots (Example D:\danu\Screenshoot) : ")

# Check and create the directory if it doesn't exist
bad_classes_dir = os.path.join(screenshot_dir, 'bad')
good_classes_dir = os.path.join(screenshot_dir, 'good')

try:
    os.makedirs(bad_classes_dir, exist_ok=True)
    os.makedirs(good_classes_dir, exist_ok=True)
except OSError as e:
    print(f"Error creating directories: {e}")
    exit()

# Dictionary to store the last detection time of each class
last_detection_time = {class_name: 0 for class_name in class_names}
last_detection_time['good_classes'] = 0  # Add entry for combined good classes
screenshot_interval = 0.3  # Interval in seconds to wait before taking another screenshot of the same class

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, size=640)
    df = results.pandas().xyxy[0]

    detected_classes = set(df['name'])

    for i in range(len(df)):
        xmin = int(df.iloc[i, 0])
        ymin = int(df.iloc[i, 1])
        xmax = int(df.iloc[i, 2])
        ymax = int(df.iloc[i, 3])
        confidence = float(df.iloc[i, 4])
        class_idx = int(df.iloc[i, 5])
        class_name = class_names[class_idx]

        # Set color based on class name
        if class_name in ['Pakaian Terbuka', 'Tidak Pakai Helm', 'Tidak Pakai Sepatu']:
            color = (0, 0, 255)  # Red color for specified classes
        else:
            color = (255, 0, 0)  # Blue color for other classes

        # Draw rectangle and text
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        text = f'{class_name} {confidence:.2f}'
        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save screenshot if bad classes are detected with confidence between 0.85 and 1.0
        current_time = time.time()
        if class_name in ['Pakaian Terbuka', 'Tidak Pakai Helm', 'Tidak Pakai Sepatu'] and 0.85 <= confidence <= 1.0:
            if current_time - last_detection_time[class_name] > screenshot_interval:
                screenshot_path = os.path.join(bad_classes_dir, f'screenshot_{class_name}_{i}_{int(current_time)}.png')
                cv2.imwrite(screenshot_path, frame.copy())  # Save a copy of the frame
                last_detection_time[class_name] = current_time

    # Save screenshot if all good classes are detected together with confidence between 0.85 and 1.0
    if {'Helm', 'Pakai Sepatu', 'Pakaian Tertutup'}.issubset(detected_classes):
        if all(df[df['name'] == cls]['confidence'].max() >= 0.85 for cls in ['Helm', 'Pakai Sepatu', 'Pakaian Tertutup']):
            if current_time - last_detection_time['good_classes'] > screenshot_interval:
                screenshot_path = os.path.join(good_classes_dir, f'screenshot_good_classes_{int(current_time)}.png')
                cv2.imwrite(screenshot_path, frame.copy())  # Save a copy of the frame
                last_detection_time['good_classes'] = current_time

    # Display the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()