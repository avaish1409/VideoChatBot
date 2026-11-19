"""Speech-to-text audio input using sounddevice and faster-whisper."""

import logging
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from video_chatbot.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger("video_chatbot.audio.stt")


def _find_working_microphone() -> Tuple[int, int]:
    """Find a functional audio input device and its native sample rate."""
    # Check default device first
    try:
        default_dev = sd.default.device[0]
        if default_dev is not None and default_dev >= 0:
            dev_info = sd.query_devices(default_dev)
            if dev_info.get("max_input_channels", 0) > 0:
                sr = int(dev_info.get("default_samplerate", 16000))
                return default_dev, sr
    except Exception:
        pass

    # Search through available input devices
    all_devices = sd.query_devices()
    for idx, dev in enumerate(all_devices):
        if dev.get("max_input_channels", 0) > 0:
            sr = int(dev.get("default_samplerate", 44100))
            try:
                # Test record a tiny buffer
                test_rec = sd.rec(100, samplerate=sr, channels=1, device=idx)
                sd.wait()
                logger.info("Selected microphone device %d (%s) at %d Hz.", idx, dev["name"], sr)
                return idx, sr
            except Exception:
                continue

    raise RuntimeError("No functional microphone device found on system.")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D float32 audio to target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    target_length = int(len(audio) * target_sr / orig_sr)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    orig_indices = np.arange(len(audio))
    target_indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(target_indices, orig_indices, audio).astype(np.float32)


class SpeechToText:
    """Offline speech transcription using faster-whisper."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        logger.info("Initializing faster-whisper model [%s] on %s...", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._device_id, self._sample_rate = _find_working_microphone()
        logger.info(
            "SpeechToText ready. Using microphone device %d at %d Hz.",
            self._device_id,
            self._sample_rate,
        )

    def listen_and_transcribe(self, duration: float = 4.0) -> str:
        """Record audio from microphone and transcribe to lowercase text.

        Returns transcribed string, or 'error' if no speech or an exception occurs,
        matching legacy audioToText() contract.
        """
        try:
            num_frames = int(duration * self._sample_rate)
            logger.info("Listening for %.1f seconds...", duration)
            recording = sd.rec(
                num_frames,
                samplerate=self._sample_rate,
                channels=1,
                device=self._device_id,
                dtype="float32",
            )
            sd.wait()

            audio = recording.flatten()

            # Clean audio data and prevent overflow
            max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
            if max_val > 1.0:
                audio = audio / max_val
            audio = np.clip(audio, -1.0, 1.0)

            # Energy check: if silence or barely audible, return error
            rms_energy = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
            if rms_energy < 0.002:
                logger.info("Audio below energy threshold (silence).")
                return "error"

            # Resample to 16kHz for Whisper
            audio_16k = _resample(audio, self._sample_rate, 16000)

            segments, _ = self.model.transcribe(
                audio_16k,
                language="en",
                beam_size=1,
                vad_filter=True,
            )

            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            full_text = " ".join(text_parts).strip().lower()

            if not full_text:
                logger.info("No speech detected by Whisper.")
                return "error"

            logger.info("Transcribed text: %s", full_text)
            return full_text

        except Exception as e:
            logger.warning("STT transcription error: %s", e)
            return "error"
