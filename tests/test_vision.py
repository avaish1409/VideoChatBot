"""Tests for vision subsystems."""

import numpy as np
import pytest
from video_chatbot.vision.emotion import FaceEmotionDetector
from video_chatbot.config import YUNET_MODEL_PATH, EMOTION_MODEL_PATH


def test_models_exist():
    assert YUNET_MODEL_PATH.exists()
    assert EMOTION_MODEL_PATH.exists()


def test_face_emotion_detector():
    detector = FaceEmotionDetector()

    # Empty frame should return neutral
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    emotion = detector.detect_emotion(empty_frame)
    assert emotion == "neutral"

    # None frame should return neutral
    assert detector.detect_emotion(None) == "neutral"
