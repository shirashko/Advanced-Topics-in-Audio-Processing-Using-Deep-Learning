#!/usr/bin/env python3
"""
Audio Processing Script

This script:
1. Converts all .ogg files to .wav format and saves them in data/rep/, data/test/, or data/train/ (resampled to 16kHz)
2. Copies all existing .wav files to data/rep/, data/test/, or data/train/ (resampled to 16kHz)
3. Removes silence/quiet times from all recordings (before and after the word)
4. Handles test_spk1_f files that may have noise at the beginning
5. All output files are resampled to 16kHz
6. Files are organized by prefix: rep_* -> data/rep/, test_* -> data/test/, train_* -> data/train/
"""

import os
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


# Configuration parameters - you can adjust these
TARGET_SAMPLE_RATE = 16000  # Target sample rate for all audio files (Hz)
SILENCE_THRESHOLD_DB = -40  # Threshold in dB for silence detection (lower = more aggressive)
FRAME_LENGTH = 2048  # Frame length for silence detection
HOP_LENGTH = 512  # Hop length for silence detection
MIN_SILENCE_DURATION_MS = 100  # Minimum duration of silence to consider (in milliseconds)
PADDING_MS = 50  # Padding to keep before and after the word (in milliseconds)

# Special handling for test_spk1_f files
TEST_SPK1_F_NOISE_DURATION_MS = 200  # Expected noise duration at beginning (in milliseconds)


def convert_ogg_to_wav(ogg_path, wav_path, sample_rate=TARGET_SAMPLE_RATE):
    """
    Convert an .ogg file to .wav format and resample to target sample rate.
    
    Args:
        ogg_path: Path to the input .ogg file
        wav_path: Path to save the output .wav file
        sample_rate: Target sample rate (default: 16000 Hz)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        # Load audio file using librosa (handles .ogg format) and resample to target rate
        audio, sr = librosa.load(ogg_path, sr=sample_rate)
        
        # Save as .wav using soundfile
        sf.write(wav_path, audio, sample_rate)
        print(f"Converted: {ogg_path} -> {wav_path} (resampled to {sample_rate}Hz)")
        return True
    except Exception as e:
        print(f"Error converting {ogg_path}: {e}")
        return False


def copy_wav_file(src_path, dst_path, sample_rate=TARGET_SAMPLE_RATE):
    """
    Copy a .wav file from source to destination and resample to target sample rate.
    
    Args:
        src_path: Source .wav file path
        dst_path: Destination .wav file path
        sample_rate: Target sample rate (default: 16000 Hz)
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Load audio file and resample to target rate
        audio, sr = librosa.load(src_path, sr=sample_rate)
        
        # Save resampled audio
        sf.write(dst_path, audio, sample_rate)
        if sr != sample_rate:
            print(f"Copied and resampled: {src_path} -> {dst_path} ({sr}Hz -> {sample_rate}Hz)")
        else:
            print(f"Copied: {src_path} -> {dst_path} (already at {sample_rate}Hz)")
        return True
    except Exception as e:
        print(f"Error copying {src_path}: {e}")
        return False


def remove_silence(audio, sample_rate, is_test_spk1_f=False):
    """
    Remove silence from the beginning and end of audio.
    Also handles special case for test_spk1_f files with noise at the beginning.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        is_test_spk1_f: Whether this is a test_spk1_f file (needs special handling)
    
    Returns:
        Trimmed audio signal
    """
    # Convert padding and min silence duration from ms to samples
    padding_samples = int(PADDING_MS * sample_rate / 1000)
    min_silence_samples = int(MIN_SILENCE_DURATION_MS * sample_rate / 1000)
    
    # Special handling for test_spk1_f files
    if is_test_spk1_f:
        # Remove noise at the beginning - use intelligent detection
        # Look for where the actual speech starts by finding sustained significant energy
        window_size = int(50 * sample_rate / 1000)  # 50ms windows
        hop_size = int(10 * sample_rate / 1000)  # 10ms hop
        energy_threshold_db = -30  # Threshold for speech detection (higher = more conservative, won't cut speech)
        noise_threshold_db = -45  # Threshold below which we consider it noise
        max_noise_duration = int(400 * sample_rate / 1000)  # Allow up to 400ms of noise removal
        
        # First pass: find where sustained speech energy starts
        # We need at least 2 consecutive windows above threshold to confirm speech
        speech_start_idx = 0
        consecutive_high_energy = 0
        required_consecutive = 2  # Need 2 windows (100ms) of high energy
        
        # Track the last position with very low energy (likely noise)
        last_low_energy_idx = 0
        
        # Search in a wider range (up to 600ms) for files with long noise
        search_range = min(len(audio) - window_size, int(600 * sample_rate / 1000))
        
        for i in range(0, search_range, hop_size):
            window = audio[i:i + window_size]
            rms_energy = np.sqrt(np.mean(window**2))
            rms_db = 20 * np.log10(rms_energy + 1e-10)
            
            # Track very low energy regions (definitely noise, below -50dB)
            if rms_db < -50:  # Very quiet, definitely noise
                last_low_energy_idx = i + window_size
            
            # Check if this window has speech-level energy (above -30dB)
            if rms_db > energy_threshold_db:
                consecutive_high_energy += 1
                if consecutive_high_energy >= required_consecutive:
                    # Found sustained speech - start a bit before for safety
                    speech_start_idx = max(0, i - (required_consecutive * hop_size))
                    break
            else:
                consecutive_high_energy = 0
        
        # If we didn't find clear speech, look for the first significant energy jump
        if speech_start_idx == 0:
            # Find the first window that's significantly above noise level
            # Look for sustained energy above -35dB (not just a spike)
            for i in range(0, search_range, hop_size):
                window = audio[i:i + window_size]
                rms_energy = np.sqrt(np.mean(window**2))
                rms_db = 20 * np.log10(rms_energy + 1e-10)
                
                # Look for energy above -35dB AND check next window too (sustained)
                if rms_db > -35:
                    # Check if next window also has good energy (sustained, not just a spike)
                    if i + window_size < len(audio):
                        next_window = audio[i + hop_size:i + hop_size + window_size]
                        next_rms = np.sqrt(np.mean(next_window**2))
                        next_rms_db = 20 * np.log10(next_rms + 1e-10)
                        if next_rms_db > -35:
                            speech_start_idx = max(0, i - hop_size)
                            break
                    else:
                        speech_start_idx = max(0, i - hop_size)
                        break
        
        # If still nothing found, use the last low energy position or max duration
        if speech_start_idx == 0:
            if last_low_energy_idx > 0:
                # Remove up to where we last saw very low energy (noise)
                speech_start_idx = min(last_low_energy_idx, max_noise_duration)
            else:
                # Fallback: remove up to max_noise_duration
                speech_start_idx = max_noise_duration
        
        # Remove the noise portion
        if speech_start_idx > 0:
            audio = audio[speech_start_idx:]
            removed_ms = int(speech_start_idx * 1000 / sample_rate)
            print(f"  Removed {removed_ms}ms of noise from beginning")
    
    # Use librosa's trim function to remove silence
    # top_db: threshold in dB below the peak for considering as silence
    trimmed_audio, index = librosa.effects.trim(
        audio,
        top_db=abs(SILENCE_THRESHOLD_DB),
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )
    
    # Add small padding before and after the word
    # For test_spk1_f files, add more padding at the beginning to ensure enough space
    if is_test_spk1_f:
        # Add extra padding for test_spk1_f files (especially file 5 which was too short)
        extra_padding_ms = 100  # Add 100ms extra before the word
        extra_padding_samples = int(extra_padding_ms * sample_rate / 1000)
        padding_start = np.zeros(padding_samples + extra_padding_samples)
    else:
        padding_start = np.zeros(padding_samples) if padding_samples > 0 else np.array([])
    
    if padding_samples > 0:
        # Add padding at the end (zeros or fade-out)
        padding_end = np.zeros(padding_samples)
        
        if len(padding_start) > 0:
            trimmed_audio = np.concatenate([padding_start, trimmed_audio, padding_end])
        else:
            trimmed_audio = np.concatenate([trimmed_audio, padding_end])
    elif len(padding_start) > 0:
        # Only start padding if specified
        trimmed_audio = np.concatenate([padding_start, trimmed_audio])
    
    return trimmed_audio


def process_audio_file(input_path, output_path, is_test_spk1_f=False, sample_rate=TARGET_SAMPLE_RATE):
    """
    Process a single audio file: load, resample to target rate, remove silence, and save.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio file
        is_test_spk1_f: Whether this is a test_spk1_f file
        sample_rate: Target sample rate (default: 16000 Hz)
    """
    try:
        # Load audio file and resample to target rate
        audio, original_sr = librosa.load(input_path, sr=sample_rate)
        
        # Remove silence
        processed_audio = remove_silence(audio, sample_rate, is_test_spk1_f=is_test_spk1_f)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed audio at target sample rate
        sf.write(output_path, processed_audio, sample_rate)
        if original_sr != sample_rate:
            print(f"Processed: {input_path} -> {output_path} (resampled from {original_sr}Hz to {sample_rate}Hz)")
        else:
            print(f"Processed: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def find_audio_files(root_dir, extensions=('.ogg', '.wav')):
    """
    Find all audio files with given extensions in the directory tree.
    
    Args:
        root_dir: Root directory to search
        extensions: Tuple of file extensions to search for
    
    Returns:
        List of file paths
    """
    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_output_path(filename, output_base_dir):
    """
    Determine the output path based on filename pattern.
    Files starting with 'rep_' go to rep/, 'test_' to test/, 'train_' to train/.
    
    Args:
        filename: Name of the audio file
        output_base_dir: Base output directory (data/)
    
    Returns:
        Path object for the output file
    """
    filename_lower = filename.lower()
    if filename_lower.startswith('rep_'):
        folder = 'rep'
    elif filename_lower.startswith('test_'):
        folder = 'test'
    elif filename_lower.startswith('train_'):
        folder = 'train'
    else:
        # Default to test if pattern not recognized
        folder = 'test'
    
    return output_base_dir / folder / filename


def main():
    """
    Main function to process all audio files.
    """
    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = data_dir  # Output directly to data directory
    
    # Create output subdirectories
    (output_dir / "rep").mkdir(exist_ok=True)
    (output_dir / "test").mkdir(exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Audio Processing Script")
    print("=" * 70)
    print(f"Source directory: {data_dir}")
    print(f"Output directory: {output_dir} (with rep/, test/, train/ subdirectories)")
    print()
    
    # Step 1: Convert all .ogg files to .wav
    print("Step 1: Converting .ogg files to .wav...")
    print("-" * 70)
    ogg_files = find_audio_files(data_dir, extensions=('.ogg',))
    
    for ogg_path in ogg_files:
        filename = os.path.basename(ogg_path)
        # Change extension to .wav
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = get_output_path(wav_filename, output_dir)
        
        convert_ogg_to_wav(ogg_path, wav_path)
    
    print(f"\nConverted {len(ogg_files)} .ogg files\n")
    
    # Step 2: Copy all existing .wav files
    print("Step 2: Copying existing .wav files...")
    print("-" * 70)
    wav_files = find_audio_files(data_dir, extensions=('.wav',))
    
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        dst_path = get_output_path(filename, output_dir)
        
        copy_wav_file(wav_path, dst_path)
    
    print(f"\nCopied {len(wav_files)} .wav files\n")
    
    # Step 3: Process all files in output directory to remove silence
    print("Step 3: Removing silence from all audio files...")
    print("-" * 70)
    all_audio_files = find_audio_files(output_dir, extensions=('.wav',))
    
    processed_count = 0
    for audio_path in all_audio_files:
        # Check if this is a test_spk1_f file
        filename = os.path.basename(audio_path)
        is_test_spk1_f = filename.startswith('test_spk1_f')
        
        # Process the file (remove silence) - overwrite in place
        if process_audio_file(audio_path, audio_path, is_test_spk1_f=is_test_spk1_f):
            processed_count += 1
    
    print(f"\nProcessed {processed_count} audio files")
    print("=" * 70)
    print("Processing complete!")
    print(f"All processed files are in: {output_dir}/rep/, {output_dir}/test/, {output_dir}/train/")


if __name__ == "__main__":
    main()

