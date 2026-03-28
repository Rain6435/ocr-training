import editdistance
from jiwer import wer as compute_wer


def character_error_rate(prediction: str, ground_truth: str) -> float:
    """
    CER = edit_distance(pred, gt) / len(gt)
    Using Levenshtein distance at character level.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return editdistance.eval(prediction, ground_truth) / len(ground_truth)


def word_error_rate(prediction: str, ground_truth: str) -> float:
    """
    WER using jiwer library (handles insertions, deletions, substitutions).
    """
    if not ground_truth.strip():
        return 0.0 if not prediction.strip() else 1.0
    return compute_wer(ground_truth, prediction)
