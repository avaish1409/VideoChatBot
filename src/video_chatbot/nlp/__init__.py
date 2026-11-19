"""NLP components for text emotion analysis and corpus-based conversation."""

from video_chatbot.nlp.corpus_bot import BotResponse, CorpusBot
from video_chatbot.nlp.text_emotion import classify_text_emotion, get_emotion_scores

__all__ = [
    "CorpusBot",
    "BotResponse",
    "classify_text_emotion",
    "get_emotion_scores",
]
