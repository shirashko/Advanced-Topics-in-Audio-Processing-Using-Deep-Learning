#!/usr/bin/env python3
import argparse
import os
import pickle as pkl
from typing import Dict, List, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from ctc import ctc_collapse, ctc_forward_logprob


LOG_ZERO = -1e9


def build_pred_matrix() -> np.ndarray:
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00
    return pred


def to_log_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-12, 1.0))


def extend_with_blanks(target: Sequence[int], blank: int) -> List[int]:
    extended = [blank]
    for token in target:
        extended.append(token)
        extended.append(blank)
    return extended


def ctc_viterbi_align(
    log_probs: np.ndarray,
    target: Sequence[int],
    blank: int,
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray, float]:
    """
    Viterbi alignment (max instead of sum). Returns:
        best_path: per-frame labels (including blanks)
        state_seq: per-frame state indices in the extended sequence
        backptr: backtrace matrix (T x L)
        scores: Viterbi score matrix (T x L)
        best_path_logprob: log-probability of the best path
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be 2D: (T, C).")
    if len(target) == 0:
        T = log_probs.shape[0]
        return [blank] * T, [0] * T, np.zeros((T, 1)), np.zeros((T, 1)), 0.0

    T, _ = log_probs.shape
    extended = extend_with_blanks(target, blank)
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

    best_path_logprob = float(scores[T - 1, end_state])
    state_seq = [end_state]
    for t in range(T - 1, 0, -1):
        state_seq.append(backptr[t, state_seq[-1]])
    state_seq.reverse()

    best_path = [extended[s] for s in state_seq]
    return best_path, state_seq, backptr, scores, best_path_logprob


def plot_prob_matrix(
    log_probs: np.ndarray,
    labels: List[str],
    save_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(log_probs.T, aspect="auto", origin="lower")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.xlabel("Time")
    plt.ylabel("Label")
    plt.title(title)
    plt.colorbar(label="Log Probability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_alignment_overlay(
    log_probs: np.ndarray,
    labels: List[str],
    best_path: List[int],
    save_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(log_probs.T, aspect="auto", origin="lower")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.xlabel("Time")
    plt.ylabel("Label")
    plt.title(title)
    plt.colorbar(label="Log Probability")
    xs = list(range(len(best_path)))
    plt.plot(xs, best_path, color="red", linewidth=2)
    plt.scatter(xs, best_path, color="red", s=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_backtrace(
    backptr: np.ndarray,
    state_seq: List[int],
    save_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(backptr.T, aspect="auto", origin="lower")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title(title)
    plt.colorbar(label="Backtrace State")
    xs = list(range(len(state_seq)))
    plt.plot(xs, state_seq, color="red", linewidth=2)
    plt.scatter(xs, state_seq, color="red", s=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_pred_example(save_dir: str) -> None:
    label_mapping = {0: "a", 1: "b", 2: "^"}
    labels = [label_mapping[i] for i in range(len(label_mapping))]
    blank = 2
    target = [0, 1, 0]  # "aba"

    pred = build_pred_matrix()
    log_probs = to_log_probs(pred)

    log_prob = ctc_forward_logprob(log_probs, target, blank=blank)
    prob = float(np.exp(log_prob))

    best_path, state_seq, backptr, _, best_logprob = ctc_viterbi_align(
        log_probs, target, blank=blank
    )
    best_prob = float(np.exp(best_logprob))
    best_path_labels = [labels[idx] for idx in best_path]
    collapsed = ctc_collapse(best_path, blank=blank)
    collapsed_labels = [labels[idx] for idx in collapsed]

    print("\n=== CTC Pred Example (aba) ===")
    print(f"Forward probability P(aba): {prob:.6f} (log={log_prob:.6f})")
    print(f"Best path labels (pre-collapse): {''.join(best_path_labels)}")
    print(f"Collapsed labels: {''.join(collapsed_labels)}")
    print(f"Best path probability: {best_prob:.6f} (log={best_logprob:.6f})")

    plot_prob_matrix(
        log_probs,
        labels,
        os.path.join(save_dir, "ctc_pred_matrix.png"),
        title="CTC Pred Matrix (log probs)",
    )
    plot_alignment_overlay(
        log_probs,
        labels,
        best_path,
        os.path.join(save_dir, "ctc_aba_alignment.png"),
        title="CTC Alignment Overlay (aba)",
    )
    plot_backtrace(
        backptr,
        state_seq,
        os.path.join(save_dir, "ctc_aba_backtrace.png"),
        title="CTC Backtrace Matrix (aba)",
    )


def guess_blank_index(label_mapping: Dict[int, str], blank_symbol: str) -> int:
    for idx, symbol in label_mapping.items():
        if symbol == blank_symbol:
            return idx
    for candidate in ("^", "<blank>", "blank", "_"):
        for idx, symbol in label_mapping.items():
            if symbol == candidate:
                return idx
    raise ValueError("Could not determine blank symbol index from label mapping.")


def text_to_ids(
    text: Sequence[str],
    symbol_to_index: Dict[str, int],
) -> List[int]:
    ids = []
    for ch in text:
        if ch in symbol_to_index:
            ids.append(symbol_to_index[ch])
        elif ch == " ":
            continue
        else:
            raise ValueError(f"Character '{ch}' not in label mapping.")
    return ids


def _get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys found: {keys}")


def run_force_align_pkl(pkl_path: str, save_dir: str, blank_symbol: str) -> None:
    data = pkl.load(open(pkl_path, "rb"))
    label_mapping = _get_key(data, "label_mapping", "label mapping")
    probs = _get_key(data, "acoustic_model_out_probs", "acoustic model out probs")
    text_to_align = _get_key(data, "text_to_align", "text to align")

    if isinstance(text_to_align, str):
        text_sequence = list(text_to_align)
    else:
        text_sequence = list(text_to_align)

    symbol_to_index = {v: k for k, v in label_mapping.items()}
    blank = guess_blank_index(label_mapping, blank_symbol)

    target_ids = text_to_ids(text_sequence, symbol_to_index)
    log_probs = to_log_probs(probs)

    best_path, state_seq, backptr, _, best_logprob = ctc_viterbi_align(
        log_probs, target_ids, blank=blank
    )
    best_prob = float(np.exp(best_logprob))
    max_idx = max(label_mapping.keys())
    labels = ["?"] * (max_idx + 1)
    for idx, symbol in label_mapping.items():
        labels[idx] = symbol
    best_path_labels = [labels[idx] for idx in best_path]
    collapsed = ctc_collapse(best_path, blank=blank)
    collapsed_labels = [labels[idx] for idx in collapsed]

    print("\n=== CTC Force Align (pkl) ===")
    print(f"text_to_align: {''.join(text_sequence)}")
    print(f"Best path labels (pre-collapse): {''.join(best_path_labels)}")
    print(f"Collapsed labels: {''.join(collapsed_labels)}")
    print(f"Best path probability: {best_prob:.6f} (log={best_logprob:.6f})")

    plot_alignment_overlay(
        log_probs,
        labels,
        best_path,
        os.path.join(save_dir, "ctc_force_align_overlay.png"),
        title="CTC Force Align Overlay (pkl)",
    )
    plot_backtrace(
        backptr,
        state_seq,
        os.path.join(save_dir, "ctc_force_align_backtrace.png"),
        title="CTC Force Align Backtrace (pkl)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CTC assignment tasks (Parts 4–7).")
    parser.add_argument(
        "--pkl-path",
        default="force_align.pkl",
        help="Path to force_align.pkl (default: force_align.pkl in repo root).",
    )
    parser.add_argument(
        "--save-dir",
        default="assets",
        help="Directory to save plots (default: assets).",
    )
    parser.add_argument(
        "--blank-symbol",
        default="^",
        help="Blank symbol for pkl label mapping (default: ^).",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    run_pred_example(args.save_dir)
    run_force_align_pkl(args.pkl_path, args.save_dir, args.blank_symbol)


if __name__ == "__main__":
    main()
