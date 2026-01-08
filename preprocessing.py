import librosa
import numpy as np
import os

# Required parameters from the assignment instructions
SAMPLE_RATE_HZ = 16000  # Resample all audio files to 16 kHz
WINDOW_SIZE_MS = 25
HOP_SIZE_MS = 10
N_MELS = 80 # Number of filter banks

# Convert time-based parameters (ms) to sample-based parameters for librosa
n_fft = int(WINDOW_SIZE_MS * SAMPLE_RATE_HZ / 1000)
hop_length = int(HOP_SIZE_MS * SAMPLE_RATE_HZ / 1000)

def extract_features(file_path):
    """Calculates Mel Spectrogram for one file."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE_HZ)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS
    )
    # Return log-scaled spectrogram for DTW analysis
    return librosa.power_to_db(mel_spec, ref=np.max)


def load_and_split_dataset(data_path):
    # Dictionaries to store the Mel Spectrograms
    dataset = {
        "representative": {},  # {word: spectrogram}
        "train": {},  # {speaker_id: {word: spectrogram}}
        "evaluation": {}  # {speaker_id: {word: spectrogram}}
    }

    # Define which speaker IDs belong to which group based on the recordings
    rep_speaker = "spk1"
    train_speakers = ["spk2", "spk3", "spk4", "spk5"]  # 2 males, 2 females
    eval_speakers = ["spk6", "spk7", "spk8", "spk9"]  # 2 males, 2 females

    for filename in os.listdir(data_path):
        if filename.endswith(".wav"):
            # Example filename format: "spk1_m_digit0.wav"
            parts = filename.replace(".wav", "").split("_")
            spk_id = parts[0]
            word = parts[2]

            file_path = os.path.join(data_path, filename)
            spec = extract_features(file_path)  # Using the function we wrote before

            # 1. Class Representative [cite: 14, 28]
            if spk_id == rep_speaker:
                dataset["representative"][word] = spec

            # 2. Training Set (4 speakers)
            elif spk_id in train_speakers:
                if spk_id not in dataset["train"]:
                    dataset["train"][spk_id] = {}
                dataset["train"][spk_id][word] = spec

            # 3. Evaluation Set (4 speakers)
            elif spk_id in eval_speakers:
                if spk_id not in dataset["evaluation"]:
                    dataset["evaluation"][spk_id] = {}
                dataset["evaluation"][spk_id][word] = spec

    return dataset