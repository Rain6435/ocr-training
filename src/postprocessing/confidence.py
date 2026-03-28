import numpy as np
import re


class ConfidenceScorer:
    # Common English character trigrams for n-gram scoring
    COMMON_TRIGRAMS = {
        "the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her",
        "ter", "hat", "tha", "ere", "ate", "his", "con", "res", "ver",
        "all", "ons", "nce", "men", "ith", "ted", "ers", "pro", "thi",
    }

    def __init__(self, dictionary_words: set = None):
        self.dictionary_words = dictionary_words or set()

    def score(self, ocr_result: dict, corrected_text: str = None) -> dict:
        """
        Aggregate confidence from multiple signals:
        1. Engine confidence (from OCR output probabilities)
        2. Spell-check ratio (% of words found in dictionary)
        3. Character-level n-gram probability

        Returns:
            {
                "confidence": float,
                "needs_review": bool,
                "breakdown": {
                    "engine_confidence": float,
                    "spell_check_ratio": float,
                    "ngram_score": float,
                }
            }
        """
        engine_conf = ocr_result.get("confidence", 0.0)
        text = ocr_result.get("text", "")

        spell_ratio = self._spell_check_ratio(text)
        ngram_score = self._ngram_score(text)

        # Weighted combination
        final_confidence = (
            0.5 * engine_conf +
            0.3 * spell_ratio +
            0.2 * ngram_score
        )

        return {
            "confidence": final_confidence,
            "needs_review": final_confidence < 0.6,
            "breakdown": {
                "engine_confidence": engine_conf,
                "spell_check_ratio": spell_ratio,
                "ngram_score": ngram_score,
            },
        }

    def _spell_check_ratio(self, text: str) -> float:
        """Fraction of words that appear to be valid English."""
        words = re.findall(r"[a-zA-Z]+", text)
        if not words:
            return 0.0

        if self.dictionary_words:
            valid = sum(1 for w in words if w.lower() in self.dictionary_words)
        else:
            # Heuristic: words with common English patterns
            valid = sum(1 for w in words if self._looks_english(w))

        return valid / len(words)

    def _looks_english(self, word: str) -> bool:
        """Heuristic check if a word looks like English."""
        w = word.lower()
        # Very short words are likely valid
        if len(w) <= 3:
            return True
        # Check for vowels (English words almost always have them)
        if not re.search(r"[aeiou]", w):
            return False
        # Check for impossible consonant clusters
        if re.search(r"[^aeiou]{5,}", w):
            return False
        return True

    def _ngram_score(self, text: str) -> float:
        """Score based on character trigram frequency."""
        text_lower = text.lower()
        if len(text_lower) < 3:
            return 0.5

        trigrams = [text_lower[i:i + 3] for i in range(len(text_lower) - 2)]
        if not trigrams:
            return 0.5

        matches = sum(1 for t in trigrams if t in self.COMMON_TRIGRAMS)
        return min(matches / max(len(trigrams) * 0.1, 1), 1.0)
