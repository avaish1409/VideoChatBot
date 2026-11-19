"""Tests for NLP components (text emotion analysis and corpus chatbot)."""

import pytest
from pathlib import Path
from video_chatbot.nlp.text_emotion import classify_text_emotion, get_emotion_scores
from video_chatbot.nlp.corpus_bot import CorpusBot, BotResponse
from video_chatbot.config import DEFAULT_SAMPLE_PATH


def test_text_emotion_classification():
    assert classify_text_emotion("I am so happy and thrilled today!") == "happy"
    assert classify_text_emotion("I feel terrible, depressed, and sad.") == "sad"
    assert classify_text_emotion("I hate this, it makes me furious and angry!") == "angry"
    assert classify_text_emotion("I am terrified and terrified of heights, so much fear.") == "fear"
    assert classify_text_emotion("That is unbelievable, wow what a shock and surprise!") == "surprise"
    assert classify_text_emotion("The notebook is placed on the desk.") == "neutral"


def test_text_emotion_negation():
    # Negation should not return happy
    res = classify_text_emotion("I am not happy today.")
    assert res != "happy"


def test_corpus_bot_training_and_retrieval():
    bot = CorpusBot(name="TestBot")
    assert DEFAULT_SAMPLE_PATH.exists()
    count = bot.train(DEFAULT_SAMPLE_PATH)
    assert count > 1000

    # Test exact match
    resp1 = bot.get_response("Hello")
    assert isinstance(resp1, BotResponse)
    assert str(resp1) == "Hi there!"

    resp2 = bot.get_response("How are you doing?")
    assert str(resp2) == "I'm doing great."

    # Test case insensitivity & punctuation
    resp3 = bot.get_response("hello!")
    assert str(resp3) == "Hi there!"


def test_corpus_bot_fallback():
    bot = CorpusBot(name="TestBot")
    bot.train(DEFAULT_SAMPLE_PATH)
    resp = bot.get_response("xyzabc123randomnonexistentprompt99")
    assert isinstance(resp, BotResponse)
    assert len(str(resp)) > 0
