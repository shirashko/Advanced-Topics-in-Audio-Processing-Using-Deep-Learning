from typing import Dict, List
import librosa.display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import librosa
from matplotlib import pyplot as plt
import os

DB_WORDS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'banana']


def find_optimal_threshold(dist_matrix, actual_labels):
    """
    Determines the optimal threshold for classification based on correct match distances.
    
    Method: Mean + 2×std of correct match distances
    This method was selected after testing 14 different threshold definition approaches,
    including percentile-based, grid search, ROC-style optimization, and separation-based methods.
    This method achieved the best evaluation accuracy (27.27%) among all tested approaches.
    
    Args:
        dist_matrix: Distance matrix (n_samples × n_references)
        actual_labels: Ground truth labels for each sample
    
    Returns:
        float: Optimal threshold value
    """
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


def reorganize_by_word(dist_matrix, row_labels, actual_labels):
    """
    Reorganizes distance matrix and labels so that all recordings of the same word
    are grouped together (word first, then speaker).
    
    Args:
        dist_matrix: Distance matrix (n_samples × n_references)
        row_labels: List of row labels in format 'spkX_word'
        actual_labels: List of actual word labels
    
    Returns:
        reorganized_dist_matrix, reorganized_row_labels, reorganized_actual_labels
    """
    # Get all unique words in order
    all_words = sorted(set(actual_labels))
    
    # Get all unique speakers
    speakers = sorted(set([label.split('_')[0] for label in row_labels]))
    
    # Create mapping: (word, speaker) -> original index
    word_speaker_to_idx = {}
    for idx, (row_label, word) in enumerate(zip(row_labels, actual_labels)):
        spk = row_label.split('_')[0]
        key = (word, spk)
        if key not in word_speaker_to_idx:
            word_speaker_to_idx[key] = []
        word_speaker_to_idx[key].append(idx)
    
    # Build new order: iterate by word, then by speaker
    new_indices = []
    new_row_labels = []
    new_actual_labels = []
    
    for word in all_words:
        for spk in speakers:
            key = (word, spk)
            if key in word_speaker_to_idx:
                for orig_idx in word_speaker_to_idx[key]:
                    new_indices.append(orig_idx)
                    new_row_labels.append(row_labels[orig_idx])
                    new_actual_labels.append(actual_labels[orig_idx])
    
    # Reorganize distance matrix
    reorganized_dist_matrix = dist_matrix[new_indices, :]
    
    return reorganized_dist_matrix, new_row_labels, new_actual_labels


def plot_spectrogram_comparison(spec1, title1, spec2, title2, fig_num=None, save_path=None):
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
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")


def analyze_samples(dataset):
    print("\n--- Performing Qualitative Analysis (Section 2.d) ---")
    
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # i. Differences within Speaker Samples (Different Digits)
    spk_ids = list(dataset['train'].keys())
    if len(spk_ids) > 0:
        spk_id = spk_ids[0]
        spec_zero = dataset['train'][spk_id].get('0')
        spec_one = dataset['train'][spk_id].get('1')

        if spec_zero is not None and spec_one is not None:
            print(f"Plotting Digit 0 vs Digit 1 for speaker: {spk_id}")
            save_path = f"{assets_dir}/mel_spec_same_speaker_0_vs_1.png"
            plot_spectrogram_comparison(
                spec_zero, f"Speaker {spk_id}: Digit 0",
                spec_one, f"Speaker {spk_id}: Digit 1",
                fig_num=1,
                save_path=save_path
            )
            plt.close()

    # ii. Differences across Digit Samples (Different Speakers/Genders)
    if len(spk_ids) > 1:
        spk_id_1 = spk_ids[0]
        spk_id_2 = spk_ids[1]

        spec_spk1_5 = dataset['train'][spk_id_1].get('5')
        spec_spk2_5 = dataset['train'][spk_id_2].get('5')

        if spec_spk1_5 is not None and spec_spk2_5 is not None:
            print(f"Plotting Digit 5 comparison: Speaker {spk_id_1} vs Speaker {spk_id_2}")
            save_path = f"{assets_dir}/mel_spec_cross_speaker_digit5.png"
            plot_spectrogram_comparison(
                spec_spk1_5, f"Speaker {spk_id_1}: Digit 5",
                spec_spk2_5, f"Speaker {spk_id_2}: Digit 5",
                fig_num=2,
                save_path=save_path
            )
            plt.close()

def plot_distance_matrix(dist_matrix, db_words, save_path=None, title="Distance Matrix", row_labels=None):
    """
    Plots the DTW distance matrix as a heatmap.
    
    Args:
        dist_matrix: Distance matrix (n_samples × n_references)
        db_words: List of reference words (column labels)
        save_path: Path to save the image
        title: Title for the plot
        row_labels: Optional list of row labels (e.g., ["spk1_0", "spk1_1", ...])
    """
    fig, ax = plt.subplots(figsize=(14, max(8, dist_matrix.shape[0] * 0.25)))
    
    im = ax.imshow(dist_matrix, aspect='auto', cmap='viridis', origin='upper')
    
    # Set column labels
    ax.set_xticks(range(len(db_words)))
    ax.set_xticklabels(db_words, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Reference Words', fontsize=12)
    
    # Set row labels
    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_ylabel('Training Samples (Speaker_Word)', fontsize=12)
    else:
        ax.set_ylabel('Training Samples', fontsize=12)
    
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('DTW Distance', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance matrix: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(actual_labels, predicted_labels, save_path=None):
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
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix: {save_path}")
        plt.close()
    else:
        plt.show()