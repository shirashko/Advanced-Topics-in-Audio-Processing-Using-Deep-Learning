from typing import Dict, List
import librosa.display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import librosa
from matplotlib import pyplot as plt

DB_WORDS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'banana']


def find_optimal_threshold(dist_matrix, actual_labels):
    correct_match_distances = []
    for i, label in enumerate(actual_labels):
        if label in DB_WORDS:
            db_idx = DB_WORDS.index(label)
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


def classify_recordings(dist_matrix, threshold=0.5):  # Added threshold param
    """
    Determines the closest word for each recording based on the minimum cost (Section 3.e).
    If the cost is too high, it is labeled as a non-digit/random word.
    """
    predicted_labels = []

    for row in dist_matrix:
        min_idx = np.argmin(row)
        min_dist = row[min_idx]

        if min_dist > threshold:
            predicted_labels.append('banana / non digit')
        else:
            predicted_labels.append(DB_WORDS[min_idx])

    return predicted_labels


def calculate_accuracy(predictions, actual_labels):
    """
    Calculates the classification accuracy (Section 3.e).
    """
    correct = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
    return (correct / len(actual_labels)) * 100 if actual_labels else 0


def get_labels_and_data(dataset: Dict, mode: str) -> List[str]:
    """Extracts ground truth labels from the dataset based on the mode."""
    labels = []
    mode_data = dataset.get(mode, {})
    # Using sorted keys ensures alignment with the distance matrix logic
    for spk in sorted(mode_data.keys()):
        for word in sorted(mode_data[spk].keys()):
            labels.append(word)
    return labels


def plot_spectrogram_comparison(spec1, title1, spec2, title2, fig_num=None):
    plt.figure(num=fig_num, figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(spec1, x_axis='time', y_axis='mel', sr=16000, hop_length=160)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title1)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spec2, x_axis='time', y_axis='mel', sr=16000, hop_length=160)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title2)

    plt.tight_layout()


def analyze_samples(dataset):
    print("\n--- Performing Qualitative Analysis (Section 2.d) ---")

    # i. Differences within Speaker Samples (Different Digits)
    spk_ids = list(dataset['train'].keys())
    if len(spk_ids) > 0:
        spk_id = spk_ids[0]
        spec_zero = dataset['train'][spk_id].get('0')
        spec_one = dataset['train'][spk_id].get('1')

        if spec_zero is not None and spec_one is not None:
            print(f"Plotting Digit 0 vs Digit 1 for speaker: {spk_id}")
            plot_spectrogram_comparison(
                spec_zero, f"Speaker {spk_id}: Digit 0",
                spec_one, f"Speaker {spk_id}: Digit 1",
                fig_num=1
            )

    # ii. Differences across Digit Samples (Different Speakers/Genders)
    if len(spk_ids) > 1:
        spk_id_1 = spk_ids[0]
        spk_id_2 = spk_ids[1]

        spec_spk1_5 = dataset['train'][spk_id_1].get('5')
        spec_spk2_5 = dataset['train'][spk_id_2].get('5')

        if spec_spk1_5 is not None and spec_spk2_5 is not None:
            print(f"Plotting Digit 5 comparison: Speaker {spk_id_1} vs Speaker {spk_id_2}")
            plot_spectrogram_comparison(
                spec_spk1_5, f"Speaker {spk_id_1}: Digit 5",
                spec_spk2_5, f"Speaker {spk_id_2}: Digit 5",
                fig_num=2
            )
    plt.show()

def plot_confusion_matrix(actual_labels, predicted_labels):
    """
    Computes and plots a confusion matrix to evaluate classification
    accuracy (over the train/validation set).
    """
    cm = confusion_matrix(actual_labels, predicted_labels, labels=DB_WORDS)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DB_WORDS)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
    cbar = ax.images[-1].colorbar
    max_val = cm.max()
    ticks = np.arange(0, max_val + 1, 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.set_label('Number of Audio Samples', rotation=270, labelpad=15, fontsize=12)
    plt.title("Confusion Matrix - Analysis", fontsize=14)
    plt.tight_layout()
    plt.show()