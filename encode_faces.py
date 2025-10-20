import face_recognition
import os
import pickle
import cv2

# Folder that contains subfolders of student images
DATASET_DIR = "dataset"  # e.g., dataset/Sainath, dataset/Aryan, etc.
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

for student_folder in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR, student_folder)
    if not os.path.isdir(path):
        continue

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(student_folder)

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encodings saved to {ENCODINGS_FILE}")
