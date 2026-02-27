"""
Configuration for the Webcam Agent Demo.
Edit PERSONS and PERSON_FACTS to match the people you want to recognise.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PHOTOS_DIR = os.path.join(PROJECT_ROOT, "reference_photos")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CAPTURES_DIR = os.path.join(PROJECT_ROOT, "captures")

FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# ── Person definitions ────────────────────────────────────────────────
# Map label ID → folder name inside reference_photos/
PERSONS = {
    0: "person1",
    1: "person2",
}

# Friendly display names
PERSON_NAMES = {
    0: "Kabeer",
    1: "Sarthak",
}

# Fun facts the agent will use in personalised greetings
PERSON_FACTS = {
    0: [
        "Kabeer is a tech enthusiast who loves building AI projects.",
        "Kabeer enjoys giving talks about agentic AI frameworks.",
        "Kabeer is a big fan of open-source LLMs.",
    ],
    1: [
        "Sarthak is an awesome collaborator who helps demo cool projects.",
        "Sarthak is always up for brainstorming creative ideas.",
        "Sarthak has great taste in music.",
    ],
}

# ── LLM / Ollama settings ────────────────────────────────────────────
LLM_MODEL = "qwen3:4b"
LLM_SERVER = "http://localhost:11434/v1"
LLM_API_KEY = "EMPTY"  # Ollama doesn't need a real key

# ── Face recognition tunables ─────────────────────────────────────────
# LBPH confidence threshold – lower is a better match.
# Predictions above this threshold will be labelled "unknown".
CONFIDENCE_THRESHOLD = 80

# ── Hardware settings ────────────────────────────────────────────────
# Usually 0 for built-in, 1 or 2 for external USB webcams.
WEBCAM_INDEX = 0
