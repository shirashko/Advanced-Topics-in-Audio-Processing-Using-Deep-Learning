import os
import logging
from typing import Dict
from preprocessing import load_and_split_dataset
from dtw_classification import build_distance_matrix
from utils import DB_WORDS, find_optimal_threshold, classify_recordings, calculate_accuracy, get_labels_and_data, \
    analyze_samples

CONFIG = {
    "DATA_DIR": "data",
    "EXPECTED_FILES": 11,
    "EXPECTED_SPEAKERS": 4,
    "MODE": "train",  # options: 'train', 'evaluation'
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

    return True


def run_asr_pipeline(mode: str = CONFIG["MODE"]):
    """Executes the DTW-based ASR classification pipeline."""
    data_path = os.path.join(os.getcwd(), CONFIG["DATA_DIR"])

    logging.info(f"Loading data from: {data_path}")
    dataset = load_and_split_dataset(data_path)

    if not validate_dataset(dataset):
        logging.warn("Dataset validation failed. Check your data directory.")

    analyze_samples(dataset)

    logging.info(f"Calculating DTW Distance Matrix (Mode: {mode})...")
    dist_matrix = build_distance_matrix(dataset[mode], dataset['representative'], DB_WORDS)

    actual_labels = get_labels_and_data(dataset, mode)

    optimal_threshold = find_optimal_threshold(dist_matrix, actual_labels)
    logging.info(f"Optimal threshold determined: {optimal_threshold:.4f}")

    predictions = classify_recordings(dist_matrix, optimal_threshold)
    accuracy = calculate_accuracy(predictions, actual_labels)

    print("\n" + "=" * 30)
    print(f"RESULTS FOR MODE: {mode.upper()}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 30)

    for i in range(min(11, len(predictions))):
        print(f"Sample {i} | Target: {actual_labels[i]:<10} | Pred: {predictions[i]}")


if __name__ == "__main__":
    try:
        run_asr_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")