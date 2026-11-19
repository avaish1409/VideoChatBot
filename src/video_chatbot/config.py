"""Configuration and path settings for VideoChatBot."""

import os
from pathlib import Path

# Base directories
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

# Resources directories
RESOURCES_DIR = ROOT_DIR / "resources"
SAMPLES_DIR = RESOURCES_DIR / "samples"
MODELS_DIR = RESOURCES_DIR / "models"

# Resource files
DEFAULT_GIF_PATH = RESOURCES_DIR / "boy-talk.gif"
DEFAULT_SAMPLE_PATH = SAMPLES_DIR / "sample0.json"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet.onnx"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion-ferplus-8.onnx"

# Vision Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FACE_INPUT_SIZE = (320, 320)
EMOTION_INPUT_SIZE = (64, 64)

# Emotion Labels (FER+ mapped to standard labels)
EMOTION_LABELS = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "angry",
    "disgust",
    "fear",
    "neutral"  # 'contempt' mapped to 'neutral'
]

# Audio Configuration
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
SILENCE_ENERGY_THRESHOLD = 0.01
SPEECH_RATE = 145  # Words per minute for pyttsx3
WHISPER_MODEL_SIZE = "tiny.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Video / UI Configuration
WINDOW_TITLE = "VideoChatBot"
TARGET_FPS = 30
FRAME_DELAY_MS = int(1000 / TARGET_FPS)
