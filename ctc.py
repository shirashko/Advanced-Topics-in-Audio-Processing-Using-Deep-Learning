from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


LOG_ZERO = -1e9


def ctc_collapse(path: Iterable[int], blank: int = 0) -> List[int]:
    """
    Collapse a CTC path by removing consecutive repeats and blanks.
    """
    collapsed = []
    prev = None
    for token in path:
        if token == prev:
            continue
        if token != blank:
            collapsed.append(token)
        prev = token
    return collapsed


def _logsumexp(values: Sequence[float]) -> float:
    values = [v for v in values if v > LOG_ZERO / 2]
    if not values:
        return LOG_ZERO
    max_val = max(values)
    return max_val + np.log(np.sum(np.exp(np.array(values) - max_val)))


def _extend_with_blanks(target: Sequence[int], blank: int) -> List[int]:
    extended = [blank]
    for token in target:
        extended.append(token)
        extended.append(blank)
    return extended


def ctc_forward_logprob(
    log_probs: np.ndarray,
    target: Sequence[int],
    blank: int = 0,
) -> float:
    """
    Compute the CTC forward log-probability for a target sequence.

    Args:
        log_probs: (T, C) array of log-probabilities over classes (including blank).
        target: Target label sequence (no blanks).
        blank: Blank label index.
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be a 2D array of shape (T, C).")
    if len(target) == 0:
        return float(np.sum(log_probs[:, blank]))

    T, _ = log_probs.shape
    extended = _extend_with_blanks(target, blank)
    L = len(extended)

    alpha = np.full((T, L), LOG_ZERO, dtype=np.float64)
    alpha[0, 0] = log_probs[0, blank]
    if L > 1:
        alpha[0, 1] = log_probs[0, extended[1]]

    for t in range(1, T):
        for s in range(L):
            candidates = [alpha[t - 1, s]]
            if s - 1 >= 0:
                candidates.append(alpha[t - 1, s - 1])
            if (
                s - 2 >= 0
                and extended[s] != blank
                and extended[s] != extended[s - 2]
            ):
                candidates.append(alpha[t - 1, s - 2])
            alpha[t, s] = log_probs[t, extended[s]] + _logsumexp(candidates)

    if L == 1:
        return float(alpha[T - 1, 0])
    return float(_logsumexp([alpha[T - 1, L - 1], alpha[T - 1, L - 2]]))


def ctc_force_align(
    log_probs: np.ndarray,
    target: Sequence[int],
    blank: int = 0,
) -> Tuple[List[int], List[int], List[Tuple[int, int, int]]]:
    """
    Compute a best-path (Viterbi) forced alignment for a target sequence.

    Returns:
        best_path: per-frame label indices (including blanks)
        collapsed: CTC-collapsed label indices
        segments: list of (label, start_frame, end_frame) for non-blank labels
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be a 2D array of shape (T, C).")
    if len(target) == 0:
        T = log_probs.shape[0]
        best_path = [blank] * T
        return best_path, [], []

    T, _ = log_probs.shape
    extended = _extend_with_blanks(target, blank)
    L = len(extended)

    scores = np.full((T, L), LOG_ZERO, dtype=np.float64)
    backptr = np.full((T, L), -1, dtype=np.int64)

    scores[0, 0] = log_probs[0, blank]
    if L > 1:
        scores[0, 1] = log_probs[0, extended[1]]
        backptr[0, 1] = 1

    for t in range(1, T):
        for s in range(L):
            candidates = [(scores[t - 1, s], s)]
            if s - 1 >= 0:
                candidates.append((scores[t - 1, s - 1], s - 1))
            if (
                s - 2 >= 0
                and extended[s] != blank
                and extended[s] != extended[s - 2]
            ):
                candidates.append((scores[t - 1, s - 2], s - 2))
            best_score, best_state = max(candidates, key=lambda x: x[0])
            scores[t, s] = log_probs[t, extended[s]] + best_score
            backptr[t, s] = best_state

    if L == 1:
        end_state = 0
    else:
        end_state = L - 1 if scores[T - 1, L - 1] >= scores[T - 1, L - 2] else L - 2

    state_seq = [end_state]
    for t in range(T - 1, 0, -1):
        state_seq.append(backptr[t, state_seq[-1]])
    state_seq.reverse()

    best_path = [extended[s] for s in state_seq]
    collapsed = ctc_collapse(best_path, blank=blank)

    segments = []
    start = None
    current_label = None
    for idx, label in enumerate(best_path):
        if label == blank:
            if current_label is not None:
                segments.append((current_label, start, idx - 1))
                current_label = None
                start = None
            continue
        if current_label is None:
            current_label = label
            start = idx
        elif label != current_label:
            segments.append((current_label, start, idx - 1))
            current_label = label
            start = idx
    if current_label is not None:
        segments.append((current_label, start, T - 1))

    return best_path, collapsed, segments