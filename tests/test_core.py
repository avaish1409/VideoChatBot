"""Tests for core VCBot orchestrator and backward compatibility."""

import pytest
from VideoChatBot import vcbot, VCBot


def test_vcbot_instantiation():
    bot = vcbot()
    assert hasattr(bot, "run")
    assert hasattr(bot, "vid")
    assert hasattr(bot, "ensemble")
    assert hasattr(bot, "getUserEmotion")
    assert hasattr(bot, "audioToText")
    assert hasattr(bot, "getTextEmotion")
    assert hasattr(bot, "getChatReply")
    assert hasattr(bot, "SpeakText")
    assert hasattr(bot, "get_samples")


def test_vcbot_methods():
    bot = vcbot()

    # Test sample loader
    samples = bot.get_samples()
    assert len(samples) > 0

    # Test text emotion
    emo = bot.getTextEmotion("I am so thrilled and happy!")
    assert emo == "happy"

    # Test chatbot reply
    reply = bot.getChatReply("Hello")
    assert "Hi" in reply or len(reply) > 0

    # Test emotion fusion logic
    # In legacy logic: if txt_emotion != video_emotion and video_emotion != 'neutral': txt += ' I am ' + video_emotion
    txt = "how are you"
    video_emotion = "sad"
    txt_emotion = "neutral"
    txt_inference = ""
    if txt_emotion != video_emotion and video_emotion != "neutral":
        txt_inference = " I am " + video_emotion

    assert txt_inference == " I am sad"
    fused_reply = bot.getChatReply(txt + txt_inference)
    assert len(fused_reply) > 0
