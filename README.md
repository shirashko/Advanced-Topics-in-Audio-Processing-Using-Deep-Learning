# Assignment 2: Automatic Speech Recognition (ASR) basics

**Course:** Advanced Topics in Audio Processing using Deep Learning 

**Environment:** Python 3.10

## Project Overview

This project implements a basic ASR pipeline, including data acquisition, feature extraction using Mel Spectrograms, and sequence alignment/classification using **Dynamic Time Warping (DTW)** and **Connectionist Temporal Classification (CTC)**.

## 1. Dataset & Preprocessing

**Data Collection:** Recorded 9 individuals (4 females, 5 males) pronouncing digits 0-9 and a random word ("banana").


**Segmentation:** Manually segmented into 11 audio files per speaker (Total: 99 files).

**Resampling:** All audio files are resampled to a consistent frequency of 16 kHz.

**Data Split:**
**Class Representative:** 1 speaker.

**Training Set:** 2 males, 2 females.

**Evaluation Set:** 2 males, 2 females.


## 2. Feature Extraction

For each audio file, we compute a Mel Spectrogram with the following parameters:
**Window size:** 25ms, **Hop size:** 10ms, **Filter banks (N_mels):** 80.

## 3. Dynamic Time Warping (DTW)
**Goal:** Align training/evaluation samples against the Class Representative's reference signals.
**Classification:** Each recording is classified based on the minimum DTW cost.
**Thresholding:** A similarity threshold is implemented to ensure the random word ("banana") is correctly labeled as a non-digit.
**Optimization:** Applied normalization (AGC) and length-based distance normalization.


## 4. CTC Algorithm (Forward & Force Alignment)
* Implementation of the CTC collapse function .
**Forward Pass:** Calculation of the sequence probability using the forward variable .
**Force Alignment:** Finding the most probable path and aligning the text to the audio signal.


## Installation & Usage

1. Clone the repository: `git clone git@github.com:shirashko/Advanced-Topics-in-Audio-Processing-Using-Deep-Learning.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`