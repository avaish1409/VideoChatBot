"""Core orchestrator for VideoChatBot."""

import logging
import os
from pathlib import Path
import threading
from typing import List, Optional

from video_chatbot.audio.stt import SpeechToText
from video_chatbot.audio.tts import SpeechSynthesizer
from video_chatbot.config import (
    CAMERA_INDEX,
    DEFAULT_GIF_PATH,
    DEFAULT_SAMPLE_PATH,
    SAMPLES_DIR,
)
from video_chatbot.nlp.corpus_bot import CorpusBot
from video_chatbot.nlp.text_emotion import classify_text_emotion
from video_chatbot.ui.avatar import AvatarDisplay
from video_chatbot.vision.camera import Camera
from video_chatbot.vision.emotion import FaceEmotionDetector

logger = logging.getLogger("video_chatbot")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VCBot:
    """Multimodal VideoChatBot.

    Coordinates real-time facial emotion recognition, speech-to-text,
    corpus-trained chatbot responses, text emotion inference, and text-to-speech
    with an animated visual avatar in a multi-threaded architecture.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        logger.info("Initializing modern VCBot subsystems...")

        # Concurrency coordination events
        self.stop_event = threading.Event()
        self.speaking_event = threading.Event()

        # Legacy flag support: flag=True (idle), flag=False (speaking)
        global flag
        flag = True
        self._lock = threading.Lock()

        # Chatbot engine
        self.chatbot = CorpusBot(name="Charlie")
        sample_files = self.get_samples()
        if sample_files:
            self.chatbot.train(*sample_files)
        else:
            logger.warning("No sample files found in %s", SAMPLES_DIR)

        # Vision subsystems
        self.camera = Camera(device_index=camera_index)
        self.vision_detector = FaceEmotionDetector()

        # Audio subsystems
        self.stt = SpeechToText()
        self.tts = SpeechSynthesizer()

        # UI Avatar subsystem
        self.avatar = AvatarDisplay(gif_path=DEFAULT_GIF_PATH)

        logger.info("VCBot successfully initialized.")

    def get_samples(self) -> List[str]:
        """Return list of JSON conversation corpus files in resources/samples."""
        if not SAMPLES_DIR.exists():
            return []
        sample_list = [
            str(SAMPLES_DIR / f)
            for f in os.listdir(SAMPLES_DIR)
            if f.endswith(".json")
        ]
        return sample_list

    def getUserEmotion(self) -> str:
        """Capture frame from camera and return user's facial emotion.

        Returns one of: 'neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear'.
        """
        try:
            success, frame = self.camera.read_frame()
            if not success or frame is None:
                logger.info("No frame received from camera. Returning neutral.")
                return "neutral"
            return self.vision_detector.detect_emotion(frame)
        except Exception as e:
            logger.warning("Error reading user emotion: %s", e)
            return "neutral"

    def audioToText(self) -> str:
        """Listen to microphone and return transcribed lowercase text.

        Returns 'error' if no speech is detected or transcription fails.
        """
        return self.stt.listen_and_transcribe(duration=4.0)

    def getTextEmotion(self, t: str) -> str:
        """Infer emotion from text string.

        Returns one of: 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'.
        """
        return classify_text_emotion(t)

    def getChatReply(self, q: str) -> str:
        """Retrieve chatbot response for the given query."""
        response = self.chatbot.get_response(q)
        return str(response)

    def vid(self, lock: Optional[threading.Lock] = None) -> None:
        """Thread 1: Display animated avatar GIF synchronized with speaking state."""
        use_lock = lock if lock is not None else self._lock
        self.avatar.run_loop(
            speaking_event=self.speaking_event,
            stop_event=self.stop_event,
            lock=use_lock,
        )

    def SpeakText(self, command: str) -> None:
        """Speak the given text via TTS."""
        self.tts.speak(command)

    def ensemble(self, lock: Optional[threading.Lock] = None) -> None:
        """Thread 2: Multimodal interaction loop."""
        txt = ""
        logger.info("Ensemble loop started.")

        try:
            # Open camera stream for ensemble loop
            self.camera.open()

            while txt != "exit" and not self.stop_event.is_set():
                # 1. Visual emotion from webcam
                video_emotion = self.getUserEmotion()
                logger.info("Video Emotion: %s", video_emotion)

                if self.stop_event.is_set():
                    break

                # 2. Prompt user: "Your Turn"
                self.SpeakText("Your Turn")

                if self.stop_event.is_set():
                    break

                # 3. Listen to microphone and transcribe speech
                txt = self.audioToText()
                logger.info("Audio To Text: %s", txt)

                if txt == "exit" or self.stop_event.is_set():
                    logger.info("Exit requested.")
                    self.stop_event.set()
                    break

                # If STT failed to detect any speech, retry next cycle
                if txt == "error":
                    logger.info("No input recognized, continuing loop...")
                    continue

                # 4. Infer text emotion
                txt_emotion = self.getTextEmotion(txt)
                logger.info("Text Emotion: %s", txt_emotion)

                # 5. Multimodal emotion fusion
                txt_inference = ""
                if txt_emotion != video_emotion and video_emotion != "neutral":
                    txt_inference = " I am " + video_emotion
                    logger.info("Added Text: %s", txt_inference)

                # 6. Chatbot response
                cbot_reply = self.getChatReply(txt + txt_inference)
                logger.info("Chatbot Reply: %s", cbot_reply)

                if self.stop_event.is_set():
                    break

                # 7. Animate avatar and speak reply
                self.speaking_event.set()
                global flag
                flag = False
                logger.info("Speaking reply...")
                self.SpeakText(cbot_reply)

                # Reset to idle avatar
                flag = True
                self.speaking_event.clear()

        except Exception as e:
            logger.error("Error in ensemble execution: %s", e)
        finally:
            self.camera.release()
            self.stop_event.set()
            logger.info("Ensemble loop terminated.")

    def run(self) -> None:
        """Launch the dual-threaded bot execution."""
        logger.info("Starting VideoChatBot dual-threaded runtime...")
        self.stop_event.clear()
        self.speaking_event.clear()

        lock = self._lock

        # Thread 1: Display GIF
        t1 = threading.Thread(target=self.vid, args=(lock,), name="AvatarUIThread")
        # Thread 2: Ensemble
        t2 = threading.Thread(target=self.ensemble, args=(lock,), name="EnsembleThread")

        t1.start()
        t2.start()

        try:
            t1.join()
            t2.join()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping threads...")
            self.stop_event.set()
            t1.join(timeout=1.0)
            t2.join(timeout=1.0)

        print("Done!")
        logger.info("Successfully completed execution, terminating vcbot!")


# Alias for backward-compatibility with original README / PyPi
vcbot = VCBot
