# Assignment 2: Automatic Speech Recognition (ASR) basics

**Course:** Advanced Topics in Audio Processing using Deep Learning 

**Environment:** Python 3.10

## Project Overview

This project implements a basic ASR pipeline, including data loading, feature extraction using Mel Spectrograms, and sequence alignment/classification using **Dynamic Time Warping (DTW)** and **Connectionist Temporal Classification (CTC)**.

You can find the drive link of the theoretical part of the assignment [here](https://docs.google.com/document/d/19fcfkTumvDoBQVct8EGvNoBGZGCHtH-_ZDBjuqMfEvk/edit?usp=sharing)


## 1. Dataset & Preprocessing

**Data Collection:** Recorded 9 individuals (4 females, 5 males) pronouncing digits 0-9 and a random word ("banana"). Overall collected 11 audio files per speaker (Total: 99 files).

**Resampling:** All audio files are resampled to a consistent frequency of 16 kHz.

**Data Split:** 
- **Class Representative:** 1 speaker (used as the reference to align with)
- **Training Set:** 2 males, 2 females.
- **Evaluation Set:** 2 males, 2 females.


## 2. Feature Extraction

For each audio file, we compute a Mel Spectrogram with the following parameters:
**Window size:** 25ms, **Hop size:** 10ms, **Filter banks (N_mels):** 80.

## 3. Dynamic Time Warping (DTW)
**Goal:** Align training/evaluation samples against the Class Representative's reference signals.
**Classification:** Each recording is classified based on the minimum DTW cost.
**Thresholding:** A similarity threshold is implemented to ensure the random word ("banana") is correctly labeled as a non-digit.
**Optimization:** Applied normalization (AGC) and length-based distance normalization.

![](assets/dtw_visual.png)

---

## 4. CTC Algorithm - Forward and Force Alignment

### Q4: CTC Collapse Function B

- Implemented in `ctc.py`: `ctc_collapse()`.
- Removes consecutive repeats and blanks.
- Example: `[^, a, a, b, ^, a] -> [a, b, a]` and `[a, b, b, a, ^] -> [a, b, a]`.

### Q5: CTC Forward Pass (alpha)

- Implemented in `ctc.py`: `ctc_forward_logprob()`.
- Computes the total probability of a target sequence by summing over all valid CTC paths using log probabilities.

### Q5a-d: Probability Matrix and P(aba)

- Label mapping: `{0: 'a', 1: 'b', 2: '^'}` where `^` is blank.
- Probability of sequence `aba`:
  - `P(aba) = 0.088800`
  - `log P(aba) = -2.421369`
- Pred matrix plot (log probabilities):

![CTC Pred Matrix](assets/ctc_pred_matrix.png)

---

## 6. CTC Force Alignment (Viterbi)

### Q6a: Adaptation for Force Alignment

- Replace the sum operator with max (Viterbi).
- Implemented in `ctc_tasks.py`: `ctc_viterbi_align()`.

### Q6b: Most Probable Path for `aba`

- Best path (before collapse): `abba^`
- Collapsed labels: `aba`
- Sequence labels: `a b b a ^`

### Q6c: Probability of the Best Path

- `P(best path) = 0.040320`
- `log P(best path) = -3.210908`

### Q6d-e: Plots

![CTC Alignment Overlay (aba)](assets/ctc_aba_alignment.png)

![CTC Backtrace Matrix (aba)](assets/ctc_aba_backtrace.png)

---

## 7. Force Alignment on force_align.pkl

### Q7a: Loaded File Contents

- Audio sampled at 16 kHz.
- Label mapping between indices and characters (alphabet + blank).
- `acoustic_model_out_probs` is a probability matrix of shape `T x 29`.
- `gt_text` and `text_to_align`.

### Q7b-d: Results for force_align.pkl

- Text to align: `then goodbye said rats they want home`
- Best path labels (pre-collapse):
```
^^^^^^^the^nn^  ^g^o^od^^^^by^e^^^^               ^s^aid      rratt^^s^^^^^                           they      ^w^anntt^^^^^^^  ^hom^e^^^^^^^^^^^^^^^^^^^^^^^
```
- Collapsed labels: `then goodbye said rats they want home`
- Best path probability:
  - `P(best path) ~= 0`
  - `log P(best path) = -68.670566`

### Q7e: Plots

![CTC Force Align Overlay](assets/ctc_force_align_overlay.png)

![CTC Force Align Backtrace](assets/ctc_force_align_backtrace.png)

---

## Summary of Results

| Metric | Value |
|--------|-------|
| P(aba) - Forward Algorithm | 0.088800 (log: -2.421369) |
| P(aba) - Best Path (Viterbi) | 0.040320 (log: -3.210908) |
| Best Path for `aba` | `abba^` -> collapsed: `aba` |
| Force Align (pkl) Best Path Prob | ~0 (log: -68.670566) |
| Force Align (pkl) Collapsed | `then goodbye said rats they want home` |

---

## Installation & Usage

1. Clone the repository: `git clone git@github.com:shirashko/Advanced-Topics-in-Audio-Processing-Using-Deep-Learning.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
4. Run CTC tasks only: `python ctc_tasks.py`
