import os
import numpy as np
from preprocessing import load_and_split_dataset
from dtw_classification import build_distance_matrix
from preprocessing import plot_spectrogram_comparison


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


def classify_recordings(dist_matrix, db_keys):
    """
    Determines the closest word for each recording based on the minimum cost (Section 3.e).
    """
    # Find the index of the minimum value in each row
    predictions = np.argmin(dist_matrix, axis=1)
    # Map indices back to word labels (e.g., index 0 -> "0", index 10 -> "banana")
    predicted_labels = [db_keys[i] for i in predictions]
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
    db_keys = list(dataset['representative'].keys())

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
        dist_matrix = build_distance_matrix(dataset['train'], dataset['representative'])

        print(f"Distance Matrix (Shape: {dist_matrix.shape}) created.")

        # Determine classification
        predictions = classify_recordings(dist_matrix, db_keys)

        # For accuracy, we extract the true labels from the training dataset
        # This assumes words are processed in the same order as in build_distance_matrix
        actual_labels = []
        for spk in dataset['train']:
            for word in sorted(dataset['train'][spk].keys()):
                actual_labels.append(word)

        accuracy = calculate_accuracy(predictions, actual_labels)

        print("\n--- Classification Results ---")
        print(f"Accuracy over Training Set: {accuracy:.2f}%")

        # Example print of first 5 predictions
        for i in range(min(5, len(predictions))):
            print(f"File {i}: Predicted='{predictions[i]}', Actual='{actual_labels[i]}'")
    else:
        print("\n No training speakers found. Please add 'train_' files to your data folder.")


if __name__ == "__main__":
    main()