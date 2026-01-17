#!/usr/bin/env python3
"""
Generate distance matrices for Question 3 with different normalization methods.
This script creates visualizations for:
- Original distance matrix (with current normalization)
- Without distance normalization (3.g.ii comparison)
- Analysis of improvements
"""

import numpy as np
from preprocessing import load_and_split_dataset
from dtw_classification import build_distance_matrix, compute_dtw_distance
from utils import DB_WORDS, plot_distance_matrix
import os

def build_distance_matrix_no_normalization(test_data, db_data, db_keys):
    """Build distance matrix without length normalization (for comparison)"""
    # Keep ordering deterministic (must match label extraction logic).
    speakers = sorted(test_data.keys())
    num_test_files = sum(len(test_data[spk]) for spk in speakers)
    dist_matrix = np.zeros((num_test_files, len(db_keys)))
    
    row_idx = 0
    for spk in speakers:
        for word in sorted(test_data[spk].keys()):
            test_spectrogram = test_data[spk][word]
            for col_idx, db_word in enumerate(db_keys):
                reference_spectrogram = db_data[db_word]
                # Use normalize=False to compare
                dist_matrix[row_idx, col_idx] = compute_dtw_distance(
                    reference_spectrogram, test_spectrogram, normalize=False
                )
            row_idx += 1
    
    return dist_matrix

def get_row_labels(dataset, mode='train', reorganize_by_word_order=False):
    """Generate row labels in format 'spkX_word' for distance matrix"""
    from utils import get_labels_and_data
    labels = []
    mode_data = dataset.get(mode, {})
    
    if reorganize_by_word_order:
        # Group by word first, then speaker
        all_words = set()
        for spk in mode_data.keys():
            all_words.update(mode_data[spk].keys())
        all_words = sorted(all_words)
        
        speakers = sorted(mode_data.keys())
        for word in all_words:
            for spk in speakers:
                if word in mode_data[spk]:
                    labels.append(f"{spk}_{word}")
    else:
        # Original order: speaker first, then word
        for spk in sorted(mode_data.keys()):
            for word in sorted(mode_data[spk].keys()):
                labels.append(f"{spk}_{word}")
    return labels

def main():
    print("=" * 70)
    print("GENERATING DISTANCE MATRICES FOR QUESTION 3")
    print("=" * 70)
    
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_and_split_dataset('data')
    
    # Build distance matrix (original order: speaker first, then word)
    print("\n1. Computing original distance matrix (with normalization)...")
    dist_matrix_original = build_distance_matrix(
        dataset['train'], dataset['representative'], DB_WORDS, normalize_dtw=True
    )
    print(f"   Shape: {dist_matrix_original.shape}")
    
    # Get row labels and actual labels in original order
    from utils import get_labels_and_data, reorganize_by_word
    row_labels_original = get_row_labels(dataset, 'train', reorganize_by_word_order=False)
    actual_labels_original = get_labels_and_data(dataset, 'train')
    
    # Reorganize by word (word first, then speaker)
    dist_matrix_original, row_labels_train, actual_labels_train = reorganize_by_word(
        dist_matrix_original, row_labels_original, actual_labels_original
    )
    
    print(f"   Mean distance: {np.mean(dist_matrix_original):.4f}")
    print(f"   Min distance: {np.min(dist_matrix_original):.4f}")
    print(f"   Max distance: {np.max(dist_matrix_original):.4f}")
    
    # Save original distance matrix visualization
    save_path_original = f"{assets_dir}/distance_matrix_original.png"
    plot_distance_matrix(
        dist_matrix_original, DB_WORDS, 
        save_path=save_path_original,
        title="Distance Matrix - Original (with normalization, organized by word)",
        row_labels=row_labels_train
    )
    
    # Distance matrix without normalization (for comparison)
    print("\n2. Computing distance matrix without length normalization...")
    dist_matrix_no_norm = build_distance_matrix_no_normalization(
        dataset['train'], dataset['representative'], DB_WORDS
    )
    
    # Reorganize by word (same order as original)
    dist_matrix_no_norm, _, _ = reorganize_by_word(
        dist_matrix_no_norm, row_labels_original, actual_labels_original
    )
    
    print(f"   Shape: {dist_matrix_no_norm.shape}")
    print(f"   Mean distance: {np.mean(dist_matrix_no_norm):.4f}")
    print(f"   Min distance: {np.min(dist_matrix_no_norm):.4f}")
    print(f"   Max distance: {np.max(dist_matrix_no_norm):.4f}")
    
    # Normalize the non-normalized matrix for fair comparison
    # Normalize by dividing by mean to make scales comparable
    dist_matrix_no_norm_scaled = dist_matrix_no_norm / np.mean(dist_matrix_no_norm) * np.mean(dist_matrix_original)
    
    save_path_no_norm = f"{assets_dir}/distance_matrix_no_length_norm.png"
    plot_distance_matrix(
        dist_matrix_no_norm_scaled, DB_WORDS,
        save_path=save_path_no_norm,
        title="Distance Matrix - Without Length Normalization (scaled for comparison, organized by word)",
        row_labels=row_labels_train
    )
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    print(f"Original (with normalization):")
    print(f"  - Mean: {np.mean(dist_matrix_original):.4f}")
    print(f"  - Std: {np.std(dist_matrix_original):.4f}")
    print(f"  - Range: [{np.min(dist_matrix_original):.4f}, {np.max(dist_matrix_original):.4f}]")
    
    print(f"\nWithout length normalization (raw):")
    print(f"  - Mean: {np.mean(dist_matrix_no_norm):.4f}")
    print(f"  - Std: {np.std(dist_matrix_no_norm):.4f}")
    print(f"  - Range: [{np.min(dist_matrix_no_norm):.4f}, {np.max(dist_matrix_no_norm):.4f}]")
    
    print(f"\nNote: Length normalization (3.g.ii) helps by:")
    print(f"  - Making distances comparable across different audio durations")
    print(f"  - Reducing bias toward longer audio files")
    print(f"  - Improving classification accuracy")
    
    print("\n" + "=" * 70)
    print("Distance matrices saved to assets/")
    print("=" * 70)

if __name__ == "__main__":
    main()

