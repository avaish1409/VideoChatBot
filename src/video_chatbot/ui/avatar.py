"""Avatar video display renderer using OpenCV and animated GIF."""

import logging
from pathlib import Path
import threading
import time
from typing import List, Optional, Union

import cv2
import imageio.v2 as imageio
import numpy as np

from video_chatbot.config import DEFAULT_GIF_PATH, FRAME_DELAY_MS, WINDOW_TITLE

logger = logging.getLogger("video_chatbot.ui.avatar")


class AvatarDisplay:
    """Renders the bot avatar GIF with idle/talking animation states."""

    def __init__(
        self,
        gif_path: Union[str, Path] = DEFAULT_GIF_PATH,
        window_title: str = WINDOW_TITLE,
        delay_ms: int = FRAME_DELAY_MS,
    ):
        self.gif_path = Path(gif_path)
        self.window_title = window_title
        self.delay_ms = delay_ms

        if not self.gif_path.exists():
            raise FileNotFoundError(f"Avatar GIF not found at {self.gif_path}")

        logger.info("Loading avatar GIF from %s...", self.gif_path)
        raw_frames = imageio.mimread(str(self.gif_path))
        self.frames: List[np.ndarray] = []
        for f in raw_frames:
            # Handle RGBA or RGB
            if len(f.shape) == 3 and f.shape[-1] == 4:
                rgb = cv2.cvtColor(f, cv2.COLOR_RGBA2RGB)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif len(f.shape) == 3 and f.shape[-1] == 3:
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            self.frames.append(bgr)

        self.num_frames = len(self.frames)
        logger.info("Loaded %d avatar animation frames.", self.num_frames)

    def run_loop(
        self,
        speaking_event: threading.Event,
        stop_event: threading.Event,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        """Run the display animation loop.

        When speaking_event is set: cycles talking frames.
        When speaking_event is clear: displays static idle frame (frame 0).
        Stops when stop_event is set or user presses 'q'.
        """
        frame_idx = 0
        logger.info("Starting avatar display window '%s'...", self.window_title)

        try:
            cv2.namedWindow(self.window_title, cv2.WINDOW_AUTOSIZE)

            while not stop_event.is_set():
                if lock is not None:
                    lock.acquire()

                is_speaking = speaking_event.is_set()

                if is_speaking:
                    frame = self.frames[frame_idx]
                    frame_idx = (frame_idx + 1) % self.num_frames
                else:
                    frame = self.frames[0]
                    frame_idx = 0

                if lock is not None:
                    lock.release()

                cv2.imshow(self.window_title, frame)

                key = cv2.waitKey(self.delay_ms) & 0xFF
                if key == ord("q"):
                    logger.info("User requested to quit from display window.")
                    stop_event.set()
                    break
                elif key == ord("r"):
                    # Dynamic state test toggle (matching legacy feature)
                    if speaking_event.is_set():
                        speaking_event.clear()
                    else:
                        speaking_event.set()
                    logger.info("Toggled speaking state to: %s", speaking_event.is_set())

                # Check if window was closed via GUI 'X' button
                try:
                    if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed by user.")
                        stop_event.set()
                        break
                except Exception:
                    pass

        finally:
            cv2.destroyAllWindows()
            logger.info("Avatar display window closed.")
