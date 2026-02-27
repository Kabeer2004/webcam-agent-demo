"""
Train the LBPH face recognition model from reference photos.

Usage:
    python train_model.py

Expects reference photos in:
    reference_photos/person1/   (images of person 1)
    reference_photos/person2/   (images of person 2)

Outputs:
    models/face_model.yml   – trained LBPH model checkpoint
    models/labels.json      – label ID → person name mapping
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

import config


def load_training_data():
    """Load face images and their labels from reference_photos/."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = []
    labels = []

    for label_id, folder_name in config.PERSONS.items():
        person_dir = os.path.join(config.REFERENCE_PHOTOS_DIR, folder_name)
        if not os.path.isdir(person_dir):
            print(f"[WARN] Directory not found: {person_dir}  – skipping.")
            continue

        image_files = [
            f
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not image_files:
            print(f"[WARN] No images found in {person_dir}  – skipping.")
            continue

        print(
            f"Processing {len(image_files)} image(s) for "
            f"{config.PERSON_NAMES[label_id]} (label {label_id}) ..."
        )

        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            pil_img = Image.open(img_path).convert("L")  # grayscale
            img_array = np.array(pil_img, "uint8")

            detected = face_cascade.detectMultiScale(
                img_array, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(detected) == 0:
                print(f"  [SKIP] No face detected in {img_file}")
                continue

            for x, y, w, h in detected:
                face_roi = img_array[y : y + h, x : x + w]
                # Resize to a uniform size for consistency
                face_roi = cv2.resize(face_roi, (200, 200))
                faces.append(face_roi)
                labels.append(label_id)
                print(f"  [OK]   Face extracted from {img_file}")

    return faces, labels


def train_and_save():
    """Train the LBPH recogniser and save the checkpoint."""
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("\n[ERROR] No faces found! Add reference photos and try again.")
        print(f"        Put images in: {config.REFERENCE_PHOTOS_DIR}")
        sys.exit(1)

    print(f"\nTraining LBPH recogniser on {len(faces)} face sample(s) ...")
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.train(faces, np.array(labels))

    # Save model checkpoint
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    recogniser.save(config.FACE_MODEL_PATH)
    print(f"Model saved to {config.FACE_MODEL_PATH}")

    # Save label mapping
    label_map = {str(k): v for k, v in config.PERSON_NAMES.items()}
    with open(config.LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Labels saved to {config.LABELS_PATH}")

    print("\n✅ Training complete! You can now run:  python agent.py")


if __name__ == "__main__":
    train_and_save()
