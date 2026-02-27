# Webcam Agent Demo

A small agentic AI demo that uses **Qwen Agent** + **Ollama** (Qwen3 4B) with two custom tools:

| Tool | Purpose |
|------|---------|
| `take_photo` | Captures a photo from the webcam |
| `recognize_face` | Identifies who is in the photo |

When you say "hi", the agent snaps a photo, recognises your face, and greets you personally with fun facts it knows about you.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama

```bash
# Install Ollama from https://ollama.com
ollama pull qwen3:4b
ollama serve          # keep running in a separate terminal
```

### 3. Add Reference Photos

Place **3–5 clear face photos** of each person into their folder:

```
reference_photos/
├── person1/    ← photos of Person 1 (e.g. you)
└── person2/    ← photos of Person 2 (e.g. your friend)
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### 4. Configure Names & Facts

Edit `config.py` to set real names and fun facts:

```python
PERSON_NAMES = {
    0: "Kabeer",
    1: "Friend",
}

PERSON_FACTS = {
    0: ["Kabeer loves building AI projects.", ...],
    1: ["Friend is an awesome collaborator.", ...],
}
```

### 5. Train the Face Model

```bash
python train_model.py
```

This saves a checkpoint to `models/` that you can copy to any machine.

### 6. Run the Agent

```bash
python agent.py
```

Type "hi" and the agent will use the webcam to identify you, then greet you personally!

## Portability

To run on another machine, copy the `models/` directory (containing `face_model.yml` and `labels.json`) to the same location. No retraining required.

## Project Structure

```
├── agent.py             # CLI chat loop (entry point)
├── tools.py             # Custom tools: take_photo, recognize_face
├── train_model.py       # Train face recognition model
├── config.py            # Names, facts, paths, LLM settings
├── requirements.txt
├── reference_photos/    # Your face images (gitignored)
│   ├── person1/
│   └── person2/
├── models/              # Trained checkpoint (gitignored)
└── captures/            # Runtime webcam captures (gitignored)
```

## Future Scope

- 🖼️ Add a local image recognition model for general object detection
- 🎤 Voice input support
- 🌐 Web UI via Qwen Agent's built-in `WebUI`
