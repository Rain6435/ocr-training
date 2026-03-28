CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,-;:!?'\"()/&@#"
)

# CTC blank is index 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
BLANK_IDX = 0
NUM_CLASSES = len(CHARS)  # +1 for blank is added in model


def encode_text(text: str) -> list[int]:
    """Encode a text string to a list of integer indices."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_indices(indices: list[int]) -> str:
    """Decode a list of integer indices back to text."""
    return "".join(IDX_TO_CHAR.get(i, "") for i in indices if i != BLANK_IDX)
