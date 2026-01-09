import librosa
import numpy as np
import os
import librosa.display
from librosa import feature

# Required parameters from the assignment instructions
SAMPLE_RATE_HZ = 16000  # Resample all audio files to 16 kHz
WINDOW_SIZE_MS = 25
HOP_SIZE_MS = 10
NUM_MEL_BINS = 80 # Number of filter banks

AUDIO_FILE_FORMAT = ".wav"

# Convert time-based parameters (ms) to sample-based parameters for librosa
# The number of samples in each analysis window
N_FFT = int(WINDOW_SIZE_MS * SAMPLE_RATE_HZ / 1000)
# The number of samples between successive frames
HOP_LENGTH = int(HOP_SIZE_MS * SAMPLE_RATE_HZ / 1000)

def extract_features(file_path, normalize=True):
    """Calculates Mel Spectrogram for one file."""
    audio_signal, sampling_rate_hz = librosa.load(file_path, sr=SAMPLE_RATE_HZ)

    if np.max(np.abs(audio_signal)) != 0:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))

    mel_spec = feature.melspectrogram(
        y=audio_signal,
        sr=sampling_rate_hz,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=NUM_MEL_BINS
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    normalized_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return normalized_mel if normalize else log_mel


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
            group, spk_id, gender, word = parts

            # later we can add gender to what we return to utilize it for analysis

            file_path = os.path.join(data_path, filename)
            spectrogram = extract_features(file_path)

            if group == "rep":
                dataset["representative"][word] = spectrogram
            elif group == "train":
                if spk_id not in dataset["train"]:
                    dataset["train"][spk_id] = {}
                dataset["train"][spk_id][word] = spectrogram
            elif group == "eval":
                if spk_id not in dataset["evaluation"]:
                    dataset["evaluation"][spk_id] = {}
                dataset["evaluation"][spk_id][word] = spectrogram

    return dataset
