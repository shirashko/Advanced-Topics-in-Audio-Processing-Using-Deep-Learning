import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa.display
from librosa import feature

# Required parameters from the assignment instructions
SAMPLE_RATE_HZ = 16000  # Resample all audio files to 16 kHz
WINDOW_SIZE_MS = 25
HOP_SIZE_MS = 10
NUM_MEL_BINS = 80 # Number of filter banks

AUDIO_FILE_FORMAT = ".wav"

# Convert time-based parameters (ms) to sample-based parameters for librosa
n_fft = int(WINDOW_SIZE_MS * SAMPLE_RATE_HZ / 1000)
hop_length = int(HOP_SIZE_MS * SAMPLE_RATE_HZ / 1000)

def extract_features(file_path):
    """Calculates Mel Spectrogram for one file."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE_HZ)
    mel_spec = feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length,
        n_mels=NUM_MEL_BINS
    )
    # Return log-scaled spectrogram for DTW analysis
    return librosa.power_to_db(mel_spec, ref=np.max)


def load_and_split_dataset(data_path):
    dataset = {
        "representative": {},  # {word: spectrogram}
        "train": {},  # {speaker_id: {word: spectrogram}}
        "evaluation": {}  # {speaker_id: {word: spectrogram}}
    }

    # Iterate through the data folder
    for filename in os.listdir(data_path):
        if filename.endswith(AUDIO_FILE_FORMAT):
            # Parsing format: [Group]_[SpeakerID]_[Gender]_[Word].wav
            # Example: rep_spk1_f_0.wav -> ['rep', 'spk1', 'f', '0']
            parts = filename.replace(AUDIO_FILE_FORMAT, "").split("_")

            # Ensure the filename has all 4 required parts to avoid errors
            if len(parts) < 4:
                print(f"Skipping malformed filename: {filename}")
                continue

            # later we can add gender to what we return to utilize it for analysis
            group, spk_id, gender, word = parts

            file_path = os.path.join(data_path, filename)

            spec = extract_features(file_path)

            if group == "rep":
                dataset["representative"][word] = spec
            elif group == "train":
                if spk_id not in dataset["train"]:
                    dataset["train"][spk_id] = {}
                dataset["train"][spk_id][word] = spec
            elif group == "eval":
                if spk_id not in dataset["evaluation"]:
                    dataset["evaluation"][spk_id] = {}
                dataset["evaluation"][spk_id][word] = spec

    return dataset

def plot_spectrogram_comparison(spec1, title1, spec2, title2):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(spec1, x_axis='time', y_axis='mel', sr=16000, hop_length=160)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title1)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spec2, x_axis='time', y_axis='mel', sr=16000, hop_length=160)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title2)

    plt.tight_layout()
    plt.show()