#!/usr/bin/env python
# coding: utf-8
"""Main entry point for VideoChatBot."""

import os
import sys

# Ensure src directory is on sys.path
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from video_chatbot.core import VCBot, vcbot

__all__ = ["VCBot", "vcbot"]

if __name__ == "__main__":
    bot = VCBot()
    bot.run()
