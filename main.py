import os
import numpy as np
from preprocessing import load_and_split_dataset
from dtw_classification import build_distance_matrix
from preprocessing import plot_spectrogram_comparison


def find_optimal_threshold(dist_matrix, actual_labels, db_keys):
    correct_match_distances = []
    for i, label in enumerate(actual_labels):
        if label in db_keys:
            db_idx = db_keys.index(label)
            correct_match_distances.append(dist_matrix[i, db_idx])

    return np.mean(correct_match_distances) + 2 * np.std(correct_match_distances)


def sanity_check(dataset):
    print("\nDisplaying Spectrogram comparison...")
    spec_zero = dataset['representative'].get('0')
    spec_banana = dataset['representative'].get('banana')

    if spec_zero is not None and spec_banana is not None:
        plot_spectrogram_comparison(
            spec_zero, "Representative: Digit 0",
            spec_banana, "Representative: Word 'Banana'"
        )
    else:
        print("Could not find '0' or 'banana' in the representative dataset.")


def classify_recordings(dist_matrix, db_keys, threshold=0.5):  # Added threshold param
    """
    Determines the closest word for each recording based on the minimum cost (Section 3.e).
    If the cost is too high, it is labeled as a non-digit/random word.
    """
    predicted_labels = []

    for row in dist_matrix:
        min_idx = np.argmin(row)
        min_dist = row[min_idx]

        if min_dist > threshold:
            predicted_labels.append('banana / non digit')  # Or "non-digit"
        else:
            predicted_labels.append(db_keys[min_idx])

    return predicted_labels


def calculate_accuracy(predictions, actual_labels):
    """
    Calculates the classification accuracy (Section 3.e).
    """
    correct = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
    return (correct / len(actual_labels)) * 100 if actual_labels else 0



def main():
    data_path = os.path.join(os.getcwd(), 'data')

    print("--- Starting ASR Project Pipeline ---")

    # 1. Preprocessing & Data Loading
    print(f"Loading data from: {data_path}...")
    dataset = load_and_split_dataset(data_path)

    rep_count = len(dataset['representative'])
    train_count = len(dataset['train'])

    # Get the reference word list from DB (with consistent order)
    db_keys = [str(i) for i in range(10)] + ['banana']

    print(f"\nDataset Summary:")
    print(f" - Representative words (DB): {rep_count}")
    print(f" - Training speakers loaded: {train_count}")

    if rep_count == 11:
        print("Representative set is complete.")
    else:
        print(f"Warning: Representative set has {rep_count}/11 files.")

    # 2. Sanity Check Visualization
    # sanity_check(dataset)

    # 3. DTW Classification (Section 3.c, 3.d, 3.e)
    if train_count > 0:
        print("\n--- Starting DTW Distance Calculation ---")

        # Build the 40x11 distance matrix
        dist_matrix = build_distance_matrix(dataset['train'], dataset['representative'], db_keys)

        # Identify the actual labels of training set
        actual_labels = []
        for spk in dataset['train']:
            for word in sorted(dataset['train'][spk].keys()):
                actual_labels.append(word)

        # Calculate the statistical threshold based on these training labels
        optimal_threshold = find_optimal_threshold(dist_matrix, actual_labels, db_keys)
        print(f"Statistically determined threshold: {optimal_threshold:.4f}")

        # Classify using the threshold
        predictions = classify_recordings(dist_matrix, db_keys, optimal_threshold)

        accuracy = calculate_accuracy(predictions, actual_labels)

        print("\n--- Classification Results ---")
        print(f"Accuracy over Training Set: {accuracy:.2f}%")

        # Example print of first 5 predictions
        for i in range(min(11, len(predictions))):
            print(f"File {i}: Predicted='{predictions[i]}', Actual='{actual_labels[i]}'")
    else:
        print("\n No training speakers found. Please add 'train_' files to your data folder.")


if __name__ == "__main__":
    main()