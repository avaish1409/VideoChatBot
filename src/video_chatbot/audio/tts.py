"""Text-to-speech synthesis using pyttsx3."""

import logging
import threading
from typing import Optional
import pyttsx3

from video_chatbot.config import SPEECH_RATE

logger = logging.getLogger("video_chatbot.audio.tts")


class SpeechSynthesizer:
    """Thread-safe text-to-speech engine using pyttsx3."""

    def __init__(self, rate: int = SPEECH_RATE):
        self.rate = rate
        self._lock = threading.Lock()
        self._engine: Optional[pyttsx3.Engine] = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize or re-initialize the TTS engine."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            logger.info("SpeechSynthesizer initialized at rate %d.", self.rate)
        except Exception as e:
            logger.error("Failed to initialize pyttsx3 engine: %s", e)
            self._engine = None

    def speak(self, text: str) -> None:
        """Speak the given text synchronously."""
        if not text or not isinstance(text, str):
            return

        with self._lock:
            try:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except ImportError:
                    pass

                if self._engine is None:
                    self._init_engine()

                if self._engine is not None:
                    logger.info("Speaking: %s", text)
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception as e:
                logger.warning("Error during TTS speak: %s", e)
                # Re-initialize engine on error to recover from COM/state issues
                self._engine = None
