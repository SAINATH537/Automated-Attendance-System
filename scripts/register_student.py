import cv2
import os

# -------------------------------
# SETTINGS
# -------------------------------
DATASET_DIR = "dataset/students"
NUM_IMAGES = 20  # number of images to capture per student

# -------------------------------
# INPUT DETAILS
# -------------------------------
student_id = input("Enter Student ID: ").strip()
student_name = input("Enter Student Name: ").strip()

# Create folder name format "101_Sainath"
folder_name = f"{student_id}_{student_name.replace(' ', '_')}"
student_path = os.path.join(DATASET_DIR, folder_name)
os.makedirs(student_path, exist_ok=True)

# -------------------------------
# START CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(f"[INFO] Capturing {NUM_IMAGES} images for {student_name}...")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))  # Resize for consistency
        cv2.imwrite(os.path.join(student_path, f"{count}.jpg"), face)

        # Draw rectangle for feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured {count}/{NUM_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Register Student", frame)

    # Break when done or if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= NUM_IMAGES:
        break

print(f"[INFO] Saved {count} images to {student_path}")
cap.release()
cv2.destroyAllWindows()
