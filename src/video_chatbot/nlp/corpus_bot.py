"""Corpus-trained conversational response engine.

Provides offline conversation retrieval trained on dataset samples
(e.g., sample0.json), replacing ChatterBot with high-speed vectorized
TF-IDF and exact/fuzzy similarity matching.
"""

import json
import logging
import math
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("video_chatbot.nlp.corpus_bot")


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def _normalize(text: str) -> str:
    """Normalize text by lowercasing and stripping punctuation and whitespace."""
    return re.sub(r"[^\w\s]", "", text).strip().lower()


class BotResponse:
    """Chatbot response wrapper maintaining backward-compatibility with ChatterBot."""

    def __init__(self, text: str, confidence: float = 1.0):
        self.text = text
        self.confidence = confidence

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<BotResponse text='{self.text}' confidence={self.confidence:.2f}>"


class CorpusBot:
    """In-memory corpus matching chatbot."""

    DEFAULT_FALLBACKS = [
        "I'm not sure I understand, could you tell me more?",
        "Interesting, tell me more about that.",
        "I see! What else is on your mind?",
        "Could you please rephrase that?"
    ]

    def __init__(self, name: str = "Charlie"):
        self.name = name
        self.prompts: List[str] = []
        self.responses: List[str] = []
        self.exact_map: Dict[str, str] = {}

        # Vectorizer state
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_matrix: Optional[np.ndarray] = None
        self._trained = False

    def train(self, *sample_paths: Union[str, Path]) -> int:
        """Load conversation pairs from JSON files and build similarity indices."""
        total_loaded = 0
        for path in sample_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning("Sample file not found: %s", path)
                continue

            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error("Error reading %s: %e", path, e)
                continue

            conversations = data.get("conversations", [])
            for conv in conversations:
                if not isinstance(conv, list) or len(conv) < 2:
                    continue
                for i in range(len(conv) - 1):
                    prompt = str(conv[i]).strip()
                    response = str(conv[i + 1]).strip()
                    if prompt and response:
                        self.prompts.append(prompt)
                        self.responses.append(response)
                        norm_prompt = _normalize(prompt)
                        if norm_prompt and norm_prompt not in self.exact_map:
                            self.exact_map[norm_prompt] = response
                        total_loaded += 1

        if self.prompts:
            self._build_index()
            self._trained = True

        logger.info(
            "CorpusBot [%s] trained on %d conversation pairs from %d files.",
            self.name,
            len(self.prompts),
            len(sample_paths)
        )
        return total_loaded

    def _build_index(self) -> None:
        """Build TF-IDF vocabulary and normalized document vectors."""
        doc_tokens: List[List[str]] = [_tokenize(p) for p in self.prompts]
        num_docs = len(doc_tokens)

        # Count document frequencies
        df: Dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        # Keep vocabulary
        self.vocab = {term: idx for idx, (term, _) in enumerate(df.items())}
        vocab_size = len(self.vocab)

        # Calculate IDF: log((N + 1) / (df + 1)) + 1
        idf_vec = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            idf_vec[idx] = math.log((num_docs + 1) / (df[term] + 1)) + 1.0
        self.idf = idf_vec

        # Build TF-IDF document vectors
        matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
        for doc_idx, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, count in tf.items():
                if t in self.vocab:
                    col_idx = self.vocab[t]
                    matrix[doc_idx, col_idx] = count * self.idf[col_idx]

        # Unit-normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_matrix = matrix / norms

    def get_response(self, query: str) -> BotResponse:
        """Find the best matching conversational response for the given query."""
        if not query or not isinstance(query, str):
            return BotResponse(self.DEFAULT_FALLBACKS[0], confidence=0.0)

        cleaned_query = query.strip()
        norm_query = _normalize(cleaned_query)

        # 1. Exact match check
        if norm_query in self.exact_map:
            return BotResponse(self.exact_map[norm_query], confidence=1.0)

        if not self._trained or self.doc_matrix is None or len(self.prompts) == 0:
            return BotResponse(self.DEFAULT_FALLBACKS[0], confidence=0.0)

        # 2. Vectorized TF-IDF Cosine Similarity
        q_tokens = _tokenize(cleaned_query)
        if not q_tokens:
            return BotResponse(self.DEFAULT_FALLBACKS[0], confidence=0.0)

        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf: Dict[str, int] = {}
        for t in q_tokens:
            tf[t] = tf.get(t, 0) + 1

        for t, count in tf.items():
            if t in self.vocab:
                idx = self.vocab[t]
                q_vec[idx] = count * self.idf[idx]

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
            similarities = self.doc_matrix @ q_vec
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            if best_score >= 0.25:
                return BotResponse(self.responses[best_idx], confidence=best_score)

        # 3. Levenshtein / SequenceMatcher fallback for short queries or typos
        best_ratio = 0.0
        best_match_idx = -1
        # Check against top sample candidates or exact matches
        for idx, prompt in enumerate(self.prompts[:500]):
            ratio = SequenceMatcher(None, norm_query, _normalize(prompt)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_idx = idx

        if best_ratio >= 0.65 and best_match_idx >= 0:
            return BotResponse(self.responses[best_match_idx], confidence=best_ratio)

        # Fallback response
        fallback = self.DEFAULT_FALLBACKS[hash(cleaned_query) % len(self.DEFAULT_FALLBACKS)]
        return BotResponse(fallback, confidence=0.1)
