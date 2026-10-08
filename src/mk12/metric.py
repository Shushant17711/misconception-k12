"""MAP@3 exactly as defined on the Evaluation tab (R1.2)."""
from collections.abc import Sequence


def ap3(pred: Sequence[str] | str, label: str, valid: set[str] | None = None) -> float:
    if isinstance(pred, str):
        pred = pred.split()
    seen, k = set(), 0
    for c in pred:
        if valid is not None and c not in valid:
            continue  # unknown codes are ignored, not penalised
        if c in seen:
            continue  # repeats skipped
        seen.add(c); k += 1
        if k > 3:
            break
        if c == label:
            return 1.0 / k
    return 0.0


def map3(preds, labels, valid=None) -> float:
    preds, labels = list(preds), list(labels)
    return sum(ap3(p, y, valid) for p, y in zip(preds, labels)) / len(labels)


def map3_probs(P, y_idx) -> float:
    """MAP@3 from a probability matrix (n, L) and integer labels."""
    import numpy as np
    top = np.argsort(-P, axis=1)[:, :3]
    hit = top == np.asarray(y_idx)[:, None]
    return float((hit / np.arange(1, 4)).sum(1).mean())
