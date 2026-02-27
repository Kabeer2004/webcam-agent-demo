"""
Custom Qwen-Agent tools: TakePhoto and RecogniseFace.
"""

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from qwen_agent.tools.base import BaseTool, register_tool

import config


# ────────────────────────────────────────────────────────────────────
# Tool 1: Take a photo with the webcam
# ────────────────────────────────────────────────────────────────────
@register_tool("take_photo")
class TakePhoto(BaseTool):
    description = (
        "Captures a photo from the computer's webcam and saves it locally. "
        "Returns the file path of the saved image. Use this tool to see who "
        "is in front of the camera."
    )
    parameters = []  # no input parameters needed

    def call(self, params: str, **kwargs) -> str:
        os.makedirs(config.CAPTURES_DIR, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return json.dumps({"error": "Could not open webcam."})

        # Let the camera warm up
        time.sleep(0.5)

        # Discard a few frames so auto-exposure can adjust
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return json.dumps({"error": "Failed to capture image from webcam."})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(config.CAPTURES_DIR, filename)
        cv2.imwrite(filepath, frame)

        return json.dumps(
            {"status": "success", "image_path": filepath, "message": f"Photo saved to {filepath}"}
        )


# ────────────────────────────────────────────────────────────────────
# Tool 2: Recognise faces in a photo
# ────────────────────────────────────────────────────────────────────
@register_tool("recognize_face")
class RecogniseFace(BaseTool):
    description = (
        "Analyses an image to recognise which person is in it. "
        "Returns the name(s) of recognised people, or 'unknown' if nobody is recognised. "
        "You must provide the path to an image file."
    )
    parameters = [
        {
            "name": "image_path",
            "type": "string",
            "description": "Absolute path to the image file to analyse.",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        # Parse params
        if isinstance(params, str):
            params = json.loads(params)
        image_path = params.get("image_path", "")

        if not os.path.isfile(image_path):
            return json.dumps({"error": f"Image not found: {image_path}"})

        # Check model exists
        if not os.path.isfile(config.FACE_MODEL_PATH):
            return json.dumps(
                {"error": "Face model not found. Run 'python train_model.py' first."}
            )
        if not os.path.isfile(config.LABELS_PATH):
            return json.dumps(
                {"error": "Labels file not found. Run 'python train_model.py' first."}
            )

        # Load model and labels
        recogniser = cv2.face.LBPHFaceRecognizer_create()
        recogniser.read(config.FACE_MODEL_PATH)

        with open(config.LABELS_PATH, "r") as f:
            label_map = json.load(f)  # {"0": "Kabeer", "1": "Friend"}

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": f"Could not read image: {image_path}"})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(detected) == 0:
            return json.dumps(
                {
                    "recognised_people": [],
                    "message": "No faces detected in the image.",
                }
            )

        results = []
        for x, y, w, h in detected:
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))
            label_id, confidence = recogniser.predict(face_roi)

            if confidence < config.CONFIDENCE_THRESHOLD:
                name = label_map.get(str(label_id), "unknown")
                results.append(
                    {"name": name, "confidence": round(confidence, 2)}
                )
            else:
                results.append(
                    {"name": "unknown", "confidence": round(confidence, 2)}
                )

        recognised_names = [r["name"] for r in results if r["name"] != "unknown"]
        if recognised_names:
            msg = f"Recognised: {', '.join(recognised_names)}"
        else:
            msg = "No known persons recognised — could be someone new."

        return json.dumps(
            {"recognised_people": results, "message": msg}
        )
