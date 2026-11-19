"""Text emotion classifier.

Provides emotion inference for text inputs, returning emotions:
'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'.
Replaces the unmaintained text2emotion library.
"""

import re
from typing import Dict

# Emotion lexicons with affective terms and stems
EMOTION_LEXICON: Dict[str, list[str]] = {
    "happy": [
        "happy", "joy", "joyful", "delighted", "glad", "great", "excellent",
        "wonderful", "love", "loving", "fantastic", "amazing", "awesome",
        "pleased", "excited", "cheerful", "blessed", "good", "smiling",
        "smile", "laugh", "laughing", "fun", "cool", "fine", "peaceful",
        "content", "yay", "haha", "brilliant", "super", "ecstatic"
    ],
    "sad": [
        "sad", "sadness", "unhappy", "depressed", "sorrow", "crying", "cry",
        "tear", "tears", "grief", "miserable", "heartbroken", "down", "gloomy",
        "lonely", "hopeless", "hurt", "hurting", "pain", "loss", "mourn",
        "despair", "bad", "disappointed", "bummed"
    ],
    "angry": [
        "angry", "mad", "furious", "rage", "irritated", "annoyed", "hate",
        "hatred", "bitter", "hostile", "outraged", "fuming", "agitated",
        "infuriated", "pissed", "jealous", "disgusted", "offensive", "offensive"
    ],
    "surprise": [
        "surprise", "surprised", "shocked", "amazed", "astonished", "stunned",
        "unexpected", "unbelievable", "wow", "whoa", "omg", "gasp", "sudden"
    ],
    "fear": [
        "fear", "afraid", "scared", "frightened", "terrified", "panic",
        "anxious", "anxiety", "worried", "nervous", "horrified", "horror",
        "phobia", "dread", "shaking", "alarmed"
    ],
    "disgust": [
        "disgust", "disgusting", "gross", "nasty", "repulsed", "sickening",
        "revolting", "yuck", "vile", "foul"
    ]
}

NEGATION_TERMS = {
    "not", "no", "never", "hardly", "barely", "scarcely", "isn't",
    "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "cannot", "can't", "couldn't", "won't", "wouldn't"
}


def get_emotion_scores(text: str) -> Dict[str, float]:
    """Calculate emotional scores across emotion categories."""
    if not text or not isinstance(text, str):
        return {k: 0.0 for k in EMOTION_LEXICON}

    tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    scores: Dict[str, float] = {k: 0.0 for k in EMOTION_LEXICON}

    negated = False
    for i, token in enumerate(tokens):
        if token in NEGATION_TERMS:
            negated = True
            continue

        matched_category = None
        for category, keywords in EMOTION_LEXICON.items():
            if token in keywords:
                matched_category = category
                break

        if matched_category:
            if negated:
                # Invert positive to negative or neutralize
                if matched_category == "happy":
                    scores["sad"] += 1.0
                elif matched_category in ("sad", "angry"):
                    scores["neutral"] = scores.get("neutral", 0.0) + 1.0
                negated = False
            else:
                scores[matched_category] += 1.0
        else:
            # Clear negation window after 2 non-matching words
            if negated and i > 0 and tokens[i - 1] not in NEGATION_TERMS:
                negated = False

    return scores


def classify_text_emotion(text: str) -> str:
    """Classify the dominant emotion in text.

    Returns one of: 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'.
    Matches legacy text2emotion output convention.
    """
    scores = get_emotion_scores(text)
    max_val = max(scores.values()) if scores else 0.0
    if max_val <= 0.0:
        return "neutral"

    # Pick the category with the highest score
    top_emotion = max(scores, key=scores.get)
    return top_emotion.lower()
