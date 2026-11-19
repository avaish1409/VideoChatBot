# VideoChatBot

VideoChatBot is an interactive multimodal video call utility that pairs an animated talking avatar with real-time facial emotion recognition, speech transcription, and corpus-trained conversation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## How It Works

VideoChatBot operates on a **dual-threaded concurrency model**:

1. **Thread 1 (UI / Video Display)**: Renders `boy-talk.gif` via OpenCV. When the bot is idle/listening, it displays a calm static avatar (frame 0). When the bot is speaking, it cycles talking animation frames at 30 FPS. Pressing `q` or closing the window cleanly exits both threads.
2. **Thread 2 (Multimodal Ensemble)**:
   - **Facial Emotion Recognition**: Captures webcam video in-memory and detects facial emotions (e.g. *happy, sad, angry, surprise, neutral*) using OpenCV YuNet face detection and ONNX Runtime FER+.
   - **Speech Prompt**: Prompts the user (`"Your Turn"`) using offline text-to-speech.
   - **Speech-to-Text**: Listens to the user's microphone and transcribes speech using `faster-whisper`.
   - **Text Emotion Inference**: Classifies emotional valence in the user's spoken words.
   - **Emotion Fusion**: If user text emotion differs from facial emotion and visual emotion isn't neutral, the input is augmented with `" I am " + video_emotion`.
   - **Conversational Corpus Retrieval**: Matches the input against the offline conversational training dataset (`resources/samples/sample0.json`) using vectorized TF-IDF similarity.
   - **Speech Synthesis**: Speaks the response via `pyttsx3` while Thread 1 animates the avatar.

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Working webcam and microphone

### Setup

Clone the repository and install requirements:

```bash
git clone https://github.com/avaish1409/VideoChatBot.git
cd VideoChatBot
pip install -r requirements.txt
```

Or install in editable mode:
```bash
pip install -e .
```

---

## Basic Usage

Run directly from the repository root:

```bash
python main.py
```

Or import as a module:

```python
from VideoChatBot import vcbot

mybot = vcbot()
mybot.run()
```

---

## Key Controls
- Press **`q`** in the video window to quit.
- Press **`r`** to manually toggle avatar speaking animation for testing.
- Say **`"exit"`** to end the conversation loop.

---

## Development & Tests

Run the test suite with pytest:

```bash
pytest tests/ -v
```

---

## License

VideoChatBot is licensed under the [GNU General Public License v3.0](LICENSE).
