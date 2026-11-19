"""Camera video capture manager."""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("video_chatbot.vision.camera")


class Camera:
    """Manages OpenCV VideoCapture hardware access."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open camera hardware capture device."""
        if self._cap is not None and self._cap.isOpened():
            return

        logger.info("Opening camera device %d...", self.device_index)
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Try default backend if DirectShow fails
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open video capture device at index {self.device_index}."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("Camera device %d successfully opened.", self.device_index)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera."""
        if self._cap is None or not self._cap.isOpened():
            self.open()

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera.")
            return False, None
        return True, frame

    def release(self) -> None:
        """Release camera resource."""
        if self._cap is not None:
            logger.info("Releasing camera device %d...", self.device_index)
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
