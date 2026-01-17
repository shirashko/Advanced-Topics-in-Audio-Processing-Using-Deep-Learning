import os
import logging
import json
from typing import Dict, List
import numpy as np
from preprocessing import load_and_split_dataset
from dtw_classification import build_distance_matrix
from ctc import ctc_forward_logprob, ctc_force_align
from ctc_tasks import main as run_ctc_tasks
from utils import DB_WORDS, find_optimal_threshold, classify_recordings, calculate_accuracy, get_labels_and_data, \
    analyze_samples, plot_confusion_matrix

CONFIG = {
    "DATA_DIR": "data",
    "EXPECTED_FILES": 11,
    "EXPECTED_SPEAKERS": 4,
    "MODE": "all",  # options: 'train', 'evaluation', 'all'
    "RUN_IMPROVEMENTS": True,  # runs baseline + improvements (Q3.g) and saves separate confusion matrices
    "RUN_CTC": True,
    "RUN_CTC_TASKS": True,
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def validate_dataset(dataset: Dict) -> bool:
    """Checks if the dataset meets the required structural specifications."""
    logging.info("Validating dataset structure...")

    # Validate Representative Set
    rep_count = len(dataset.get('representative', {}))
    if rep_count != CONFIG["EXPECTED_FILES"]:
        logging.warning(f"Representative set mismatch: {rep_count}/{CONFIG['EXPECTED_FILES']}")
        return False

    # Validate Split Groups
    for group in ['train', 'evaluation']:
        spk_count = len(dataset.get(group, {}))
        if spk_count != CONFIG["EXPECTED_SPEAKERS"]:
            logging.warning(f"{group.capitalize()} speaker mismatch: {spk_count}/{CONFIG['EXPECTED_SPEAKERS']}")
            return False
        for speaker in dataset.get(group, {}):
            file_count = len(dataset[group][speaker])
            if file_count != CONFIG["EXPECTED_FILES"]:
                logging.warning(f"{group.capitalize()} {speaker} file mismatch: {file_count}/{CONFIG['EXPECTED_FILES']}")
                return False

    return True


def _ensure_assets_dir() -> str:
    assets_dir = os.path.join(os.getcwd(), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    return assets_dir


def _evaluate_mode(
    dataset: Dict,
    mode: str,
    optimal_threshold: float,
    *,
    config_name: str,
    normalize_dtw: bool,
) -> Dict:
    logging.info(f"Calculating DTW Distance Matrix (Mode: {mode})...")
    dist_matrix = build_distance_matrix(
        dataset[mode], dataset["representative"], DB_WORDS, normalize_dtw=normalize_dtw
    )
    actual_labels = get_labels_and_data(dataset, mode)

    predictions = classify_recordings(dist_matrix, optimal_threshold)
    accuracy = calculate_accuracy(predictions, actual_labels)

    print("\n" + "=" * 30)
    print(f"RESULTS FOR MODE: {mode.upper()} | CONFIG: {config_name}")
    print(f"Threshold (from TRAIN): {optimal_threshold:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 30)

    for i in range(min(11, len(predictions))):
        print(f"Sample {i} | Target: {actual_labels[i]:<10} | Pred: {predictions[i]}")

    assets_dir = _ensure_assets_dir()

    # Save confusion matrix
    confusion_matrix_path = os.path.join(assets_dir, f"confusion_matrix_{config_name}_{mode}.png")
    plot_confusion_matrix(actual_labels, predictions, save_path=confusion_matrix_path)

    # Also write legacy/short filenames for the "final" config so there's no ambiguity
    # about what confusion_matrix_{mode}.png refers to.
    if config_name in ("final", "improvement_1plus2_audio_plus_distance_norm"):
        legacy_confusion_matrix_path = os.path.join(assets_dir, f"confusion_matrix_{mode}.png")
        plot_confusion_matrix(actual_labels, predictions, save_path=legacy_confusion_matrix_path)

    # Save a small machine-readable results file
    results = {
        "config": config_name,
        "mode": mode,
        "threshold_from_train": float(optimal_threshold),
        "accuracy_percent": float(accuracy),
        "num_samples": int(len(actual_labels)),
    }
    with open(os.path.join(assets_dir, f"results_{config_name}_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def run_asr_pipeline(mode: str = CONFIG["MODE"]):
    data_path = os.path.join(os.getcwd(), CONFIG["DATA_DIR"])

    logging.info(f"Loading data from: {data_path}")
    assets_dir = _ensure_assets_dir()

    modes = ["train", "evaluation"] if mode == "all" else [mode]

    if CONFIG["RUN_IMPROVEMENTS"]:
        # Q3.g configs: baseline, improvement 1 (audio norm), improvement 1+2 (audio+distance norm)
        configs = [
            ("baseline", dict(audio_normalize=False, normalize_dtw=False)),
            ("improvement_1_audio_norm", dict(audio_normalize=True, normalize_dtw=False)),
            ("improvement_1plus2_audio_plus_distance_norm", dict(audio_normalize=True, normalize_dtw=True)),
        ]
    else:
        # Single "final" config (current best): audio + distance normalization.
        configs = [
            ("final", dict(audio_normalize=True, normalize_dtw=True)),
        ]

    summary = []

    for config_name, cfg in configs:
        logging.info(f"\n=== Running config: {config_name} ===")
        dataset = load_and_split_dataset(data_path, audio_normalize=cfg["audio_normalize"])

        if not validate_dataset(dataset):
            logging.error("Dataset validation failed. Check your data directory.")
            raise ValueError("Dataset validation failed. Check your data directory.")

        # Keep qualitative plots (Q2.d) generated once to avoid overwriting / duplication.
        if config_name == configs[0][0]:
            analyze_samples(dataset)

        # IMPORTANT (Assignment 2, DTW §3.e–f): compute the threshold ONLY from training
        # data, and then apply that threshold to both training and evaluation sets.
        logging.info("Calculating DTW Distance Matrix (Training, for threshold)...")
        train_dist_matrix = build_distance_matrix(
            dataset["train"],
            dataset["representative"],
            DB_WORDS,
            normalize_dtw=cfg["normalize_dtw"],
        )
        train_labels = get_labels_and_data(dataset, "train")
        optimal_threshold = find_optimal_threshold(train_dist_matrix, train_labels)
        logging.info(f"Optimal threshold determined from TRAIN: {optimal_threshold:.4f}")

        for m in modes:
            if m not in dataset:
                raise ValueError(f"Unknown mode '{m}'. Expected one of: train, evaluation, all.")
            summary.append(
                _evaluate_mode(
                    dataset,
                    m,
                    optimal_threshold,
                    config_name=config_name,
                    normalize_dtw=cfg["normalize_dtw"],
                )
            )

    # Save combined summary
    with open(os.path.join(assets_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if CONFIG["RUN_CTC"]:
        run_ctc_demo()
    if CONFIG["RUN_CTC_TASKS"]:
        run_ctc_tasks()


def _log_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    max_val = np.max(scores, axis=axis, keepdims=True)
    stabilized = scores - max_val
    log_sum_exp = np.log(np.sum(np.exp(stabilized), axis=axis, keepdims=True))
    return stabilized - log_sum_exp


def _build_synthetic_ctc_log_probs(
    target_ids: List[int],
    num_classes: int,
    blank_id: int = 0,
) -> np.ndarray:
    """
    Build a small synthetic (T, C) log-probability matrix for CTC demo purposes.
    """
    pattern = [blank_id]
    for token in target_ids:
        pattern.extend([token, token, blank_id])
    T = len(pattern)

    scores = np.full((T, num_classes), -5.0, dtype=np.float64)
    for t, token in enumerate(pattern):
        scores[t, token] = 2.0
    return _log_softmax(scores, axis=1)


def run_ctc_demo():
    blank_id = 0
    label_to_id = {label: idx + 1 for idx, label in enumerate(DB_WORDS)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    id_to_label[blank_id] = "<blank>"

    target_labels = ["1", "2", "3"]
    target_ids = [label_to_id[label] for label in target_labels]
    log_probs = _build_synthetic_ctc_log_probs(
        target_ids=target_ids,
        num_classes=len(DB_WORDS) + 1,
        blank_id=blank_id,
    )

    log_prob = ctc_forward_logprob(log_probs, target_ids, blank=blank_id)
    best_path, collapsed, segments = ctc_force_align(log_probs, target_ids, blank=blank_id)

    decoded_labels = [id_to_label[label_id] for label_id in collapsed]
    alignment_segments = [
        (id_to_label[label_id], start, end) for label_id, start, end in segments
    ]

    print("\n" + "=" * 30)
    print("CTC DEMO RESULTS")
    print(f"Target labels: {target_labels}")
    print(f"CTC log-probability: {log_prob:.4f}")
    print(f"Best-path collapsed labels: {decoded_labels}")
    print(f"Alignment segments (label, start, end): {alignment_segments}")
    print("=" * 30)


if __name__ == "__main__":
    try:
        run_asr_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")