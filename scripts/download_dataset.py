# scripts/download_dataset.py
from sklearn.datasets import fetch_lfw_people
import os, shutil

# Download LFW dataset (a few thousand labeled face images)
print("[INFO] Downloading LFW dataset... This may take a minute.")
lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.5, color=True)

base_dir = "dataset/students"
os.makedirs(base_dir, exist_ok=True)

# We'll create only a few "student" folders for simplicity
for i, name in enumerate(lfw.target_names[:3]):  # Take first 3 persons
    person_dir = os.path.join(base_dir, f"{100+i}_{name.replace(' ', '_')}")
    os.makedirs(person_dir, exist_ok=True)

# Save 5 images per person
print("[INFO] Saving sample images...")
for idx, (image, label) in enumerate(zip(lfw.images, lfw.target)):
    if label >= 3:  # only first 3 people
        continue
    person_dir = os.path.join(base_dir, f"{100+label}_{lfw.target_names[label].replace(' ', '_')}")
    cv2_path = os.path.join(person_dir, f"{idx}.jpg")

    from PIL import Image
    img = Image.fromarray((image * 255).astype("uint8"))
    img.save(cv2_path)

print("[INFO] Dataset ready in dataset/students/")
