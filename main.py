import os
from preprocessing import load_and_split_dataset

def sanity_check(dataset):
    print("\nDisplaying Spectrogram comparison...")
    # Access the spectrograms from the representative dictionary
    # The keys '0' and 'banana' match your filenames: rep_spk1_f_0.m4a and rep_spk1_f_banana.m4a
    spec_zero = dataset['representative'].get('0')
    spec_banana = dataset['representative'].get('banana')

    if spec_zero is not None and spec_banana is not None:
        from preprocessing import plot_spectrogram_comparison
        plot_spectrogram_comparison(
            spec_zero, "Representative: Digit 0",
            spec_banana, "Representative: Word 'Banana'"
        )
    else:
        print("Could not find '0' or 'banana' in the representative dataset.")


def main():
    data_path = os.path.join(os.getcwd(), 'data')

    print("--- Starting ASR Project Pipeline ---")

    # 2. Load and Preprocess Dataset
    # This will extract Mel Spectrograms (25ms window, 10ms hop, 80 filters)
    print(f"Loading data from: {data_path}...")
    dataset = load_and_split_dataset(data_path)

    # 3. Sanity Check / Data Verification
    # Verify the split requirements (2 males, 2 females per set)
    rep_count = len(dataset['representative'])
    train_count = len(dataset['train'])
    eval_count = len(dataset['evaluation'])

    print(f"\nDataset Summary:")
    print(f" - Representative words loaded: {rep_count}")
    print(f" - Training speakers loaded: {train_count}")
    print(f" - Evaluation speakers loaded: {eval_count}")

    # Check if we have the required 11 files for the representative
    if rep_count == 11:
        print("Representative set is complete (0-9 + banana).")
    else:
        print(f"Warning: Representative set has {rep_count}/11 files.")

    # 4. Visualization for sanity check of the input preprocessing
    sanity_check(dataset)

if __name__ == "__main__":
    main()