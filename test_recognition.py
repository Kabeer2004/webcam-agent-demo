"""
Test script to verify end-to-end face recognition:
1. Snaps a photo from the webcam (using config.WEBCAM_INDEX)
2. Runs the LBPH model on the captured photo
3. Displays the result

Usage:
    python test_recognition.py
"""

import cv2
import os
import json
import time
from datetime import datetime
import config

def test_recognition():
    print(f"--- Face Recognition Test ---")
    
    # 1. Snap photo
    print(f"Opening webcam (index {config.WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {config.WEBCAM_INDEX}.")
        return

    print("Capturing in 3 seconds... (Get ready!)")
    time.sleep(3)
    
    # Warmup
    for _ in range(5): cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to capture frame.")
        return

    # 2. Save capture
    os.makedirs(config.CAPTURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(config.CAPTURES_DIR, f"test_rec_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Captured photo saved to: {image_path}")

    # 3. Load Model
    if not os.path.isfile(config.FACE_MODEL_PATH):
        print(f"[ERROR] Model not found at {config.FACE_MODEL_PATH}")
        print("Please run 'python train_model.py' first.")
        return

    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.read(config.FACE_MODEL_PATH)

    with open(config.LABELS_PATH, "r") as f:
        label_map = json.load(f)

    # 4. Process Recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    detected = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(detected) == 0:
        print("\n[RESULT] No faces detected in the photo.")
        cv2.imshow("Test Result - NO FACES", frame)
    else:
        print(f"\n[RESULT] Found {len(detected)} face(s):")
        for (x, y, w, h) in detected:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            label_id, confidence = recogniser.predict(face_roi)

            name = "unknown"
            if confidence < config.CONFIDENCE_THRESHOLD:
                name = label_map.get(str(label_id), "unknown")
            
            print(f"  - {name} (Confidence: {confidence:.2f})")
            
            # Draw on frame
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Test Result - Recognition", frame)

    print("\nPress any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_recognition()
