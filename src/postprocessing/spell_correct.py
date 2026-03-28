import os
from symspellpy import SymSpell, Verbosity


class SpellCorrector:
    def __init__(
        self,
        dictionary_path: str = "data/dictionaries/en_dict.txt",
        historical_dict_path: str = "data/dictionaries/historical_en.txt",
        max_edit_distance: int = 2,
    ):
        """
        Initialize SymSpell with modern and optionally historical dictionaries.
        """
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        self.max_edit_distance = max_edit_distance

        # Load main dictionary
        if os.path.exists(dictionary_path):
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        else:
            # Use built-in dictionary as fallback
            import pkg_resources
            dict_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

        # Load historical dictionary if available
        if historical_dict_path and os.path.exists(historical_dict_path):
            self.sym_spell.load_dictionary(historical_dict_path, term_index=0, count_index=1)

    def correct(self, text: str, max_edit_distance: int = None) -> dict:
        """
        Correct spelling errors in OCR output.

        Returns:
            {
                "original": str,
                "corrected": str,
                "corrections": [{"word": str, "corrected": str, "distance": int}, ...],
                "num_corrections": int,
            }
        """
        if max_edit_distance is None:
            max_edit_distance = self.max_edit_distance

        words = text.split()
        corrected_words = []
        corrections = []

        for word in words:
            # Skip short words and numbers
            if len(word) <= 2 or word.isdigit():
                corrected_words.append(word)
                continue

            # Strip punctuation for lookup, preserve for output
            stripped = word.strip(".,;:!?\"'()-/")
            if not stripped:
                corrected_words.append(word)
                continue

            suggestions = self.sym_spell.lookup(
                stripped, Verbosity.CLOSEST, max_edit_distance=max_edit_distance
            )

            if suggestions and suggestions[0].distance > 0:
                corrected = suggestions[0].term
                # Preserve original casing pattern
                if stripped[0].isupper():
                    corrected = corrected.capitalize()
                if stripped.isupper():
                    corrected = corrected.upper()

                # Restore punctuation
                prefix = word[:len(word) - len(word.lstrip(".,;:!?\"'()-/"))]
                suffix = word[len(word.rstrip(".,;:!?\"'()-/")):]
                corrected_word = prefix + corrected + suffix

                corrected_words.append(corrected_word)
                corrections.append({
                    "word": word,
                    "corrected": corrected_word,
                    "distance": suggestions[0].distance,
                })
            else:
                corrected_words.append(word)

        return {
            "original": text,
            "corrected": " ".join(corrected_words),
            "corrections": corrections,
            "num_corrections": len(corrections),
        }
