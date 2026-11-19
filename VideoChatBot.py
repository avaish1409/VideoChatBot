"""Backward compatibility module for 'from VideoChatBot import vcbot'."""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from video_chatbot.core import VCBot, vcbot

__all__ = ["VCBot", "vcbot"]
