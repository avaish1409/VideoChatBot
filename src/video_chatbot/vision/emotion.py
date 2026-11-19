"""Facial detection and emotion classification using OpenCV YuNet and ONNX Runtime."""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

from video_chatbot.config import (
    EMOTION_INPUT_SIZE,
    EMOTION_LABELS,
    EMOTION_MODEL_PATH,
    FACE_INPUT_SIZE,
    YUNET_MODEL_PATH,
)

logger = logging.getLogger("video_chatbot.vision.emotion")


class FaceEmotionDetector:
    """Detects human faces and classifies their facial emotion in real-time."""

    def __init__(
        self,
        yunet_path: Union[str, Path] = YUNET_MODEL_PATH,
        emotion_model_path: Union[str, Path] = EMOTION_MODEL_PATH,
    ):
        self.yunet_path = Path(yunet_path)
        self.emotion_model_path = Path(emotion_model_path)

        # Ensure model directory exists
        self.yunet_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.yunet_path.exists():
            logger.info("YuNet model not found locally. Downloading from Hugging Face...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="opencv/face_detection_yunet",
                filename="face_detection_yunet_2023mar.onnx",
            )
            import shutil
            shutil.copyfile(downloaded, str(self.yunet_path))

        if not self.emotion_model_path.exists():
            logger.info("Emotion model not found locally. Downloading from Hugging Face...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="onnxmodelzoo/emotion-ferplus-8",
                filename="emotion-ferplus-8.onnx",
            )
            import shutil
            shutil.copyfile(downloaded, str(self.emotion_model_path))

        logger.info("Initializing YuNet face detector and FER+ ONNX model...")
        self.face_detector = cv2.FaceDetectorYN_create(
            str(self.yunet_path),
            "",
            FACE_INPUT_SIZE,
            score_threshold=0.5,
            nms_threshold=0.3,
        )

        # ONNX Runtime session for emotion classification
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.emotion_session = ort.InferenceSession(
            str(self.emotion_model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        self._input_name = self.emotion_session.get_inputs()[0].name
        self._output_name = self.emotion_session.get_outputs()[0].name
        logger.info("FaceEmotionDetector successfully initialized.")

    def detect_emotion(self, frame: np.ndarray) -> str:
        """Extract user facial emotion from a single video frame.

        Returns one of: 'neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear'.
        Returns 'neutral' if no face is detected, matching legacy behavior.
        """
        if frame is None or frame.size == 0:
            logger.info("Empty frame received. Returning neutral.")
            return "neutral"

        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(frame)

        if faces is None or len(faces) == 0:
            logger.info("No face detected")
            return "neutral"

        # Select face with largest area or highest confidence score
        best_face = faces[0]
        if len(faces) > 1:
            best_face = max(faces, key=lambda f: f[2] * f[3])

        # Bounding box coordinates: [x, y, w, h]
        x, y, box_w, box_h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])

        # Clamp bounding box inside image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + box_w)
        y2 = min(h, y + box_h)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            logger.info("Detected face area too small.")
            return "neutral"

        face_roi = frame[y1:y2, x1:x2]

        # Convert to grayscale and resize to (64, 64) for FER+ model
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, EMOTION_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        normalized_face = resized_face.astype(np.float32)

        # Reshape to (1, 1, 64, 64)
        input_tensor = np.expand_dims(np.expand_dims(normalized_face, axis=0), axis=0)

        # Run ONNX inference
        outputs = self.emotion_session.run([self._output_name], {self._input_name: input_tensor})
        logits = outputs[0][0]

        top_idx = int(np.argmax(logits))
        detected_emotion = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else "neutral"

        logger.info("Inferred facial emotion: %s (idx: %d)", detected_emotion, top_idx)
        return detected_emotion
