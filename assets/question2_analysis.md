# Question 2.d: Mel Spectrogram Analysis
## Analysis of Differences in Mel Spectrograms

## i. Differences within Speaker Samples (Different Digits)

### Comparison: Digit 0 vs Digit 1 (Speaker spk1)

**Image:** `assets/mel_spec_same_speaker_0_vs_1_spk1.png`

- **Duration:** Digit 0: 1.61s, Digit 1: 0.68s
- **Energy:** Digit 0: 1.0052, Digit 1: 1.0051
- **Observations:** 
  - The spectrograms show distinct frequency patterns for different digits.
  - Each digit has unique formant structures and temporal characteristics.
  - The mel-scale representation captures perceptual differences between phonemes.

### Comparison: Digit 0 vs Digit 5 (Speaker spk1)

**Image:** `assets/mel_spec_same_speaker_0_vs_5_spk1.png`

- **Duration:** Digit 0: 1.61s, Digit 5: 0.62s
- **Energy:** Digit 0: 1.0052, Digit 5: 1.0051
- **Observations:** 
  - The spectrograms show distinct frequency patterns for different digits.
  - Each digit has unique formant structures and temporal characteristics.
  - The mel-scale representation captures perceptual differences between phonemes.

### Comparison: Digit 1 vs Digit 9 (Speaker spk1)

**Image:** `assets/mel_spec_same_speaker_1_vs_9_spk1.png`

- **Duration:** Digit 1: 0.68s, Digit 9: 0.65s
- **Energy:** Digit 1: 1.0051, Digit 9: 1.0051
- **Observations:** 
  - The spectrograms show distinct frequency patterns for different digits.
  - Each digit has unique formant structures and temporal characteristics.
  - The mel-scale representation captures perceptual differences between phonemes.

## ii. Differences across Digit Samples (Different Speakers/Genders)

### Comparison: Digit 0 - Speaker spk1 vs Speaker spk2

**Image:** `assets/mel_spec_cross_speaker_digit0_spk1_vs_spk2.png`

- **Duration:** Speaker spk1: 1.61s, Speaker spk2: 0.87s
- **Energy:** Speaker spk1: 1.0052, Speaker spk2: 1.0051
- **Frequency Centroid:** Speaker spk1: 38.73, Speaker spk2: 38.21
- **Observations:** 
  - Different speakers show variations in formant frequencies due to vocal tract differences.
  - Gender differences are visible in the frequency distribution (males typically have lower formants).
  - Speaking rate and articulation style vary between individuals.
  - The overall spectral envelope shape is preserved across speakers for the same digit.

### Comparison: Digit 5 - Speaker spk1 vs Speaker spk2

**Image:** `assets/mel_spec_cross_speaker_digit5_spk1_vs_spk2.png`

- **Duration:** Speaker spk1: 0.62s, Speaker spk2: 0.84s
- **Energy:** Speaker spk1: 1.0051, Speaker spk2: 1.0050
- **Frequency Centroid:** Speaker spk1: 38.38, Speaker spk2: 38.58
- **Observations:** 
  - Different speakers show variations in formant frequencies due to vocal tract differences.
  - Gender differences are visible in the frequency distribution (males typically have lower formants).
  - Speaking rate and articulation style vary between individuals.
  - The overall spectral envelope shape is preserved across speakers for the same digit.

### Comparison: Digit 9 - Speaker spk1 vs Speaker spk2

**Image:** `assets/mel_spec_cross_speaker_digit9_spk1_vs_spk2.png`

- **Duration:** Speaker spk1: 0.65s, Speaker spk2: 0.97s
- **Energy:** Speaker spk1: 1.0051, Speaker spk2: 1.0051
- **Frequency Centroid:** Speaker spk1: 38.40, Speaker spk2: 38.26
- **Observations:** 
  - Different speakers show variations in formant frequencies due to vocal tract differences.
  - Gender differences are visible in the frequency distribution (males typically have lower formants).
  - Speaking rate and articulation style vary between individuals.
  - The overall spectral envelope shape is preserved across speakers for the same digit.

### Gender Comparison: Male vs Female (Digit 5)

**Image:** `assets/mel_spec_gender_comparison_digit5.png`

**Key Differences:**
- **Formant Frequencies:** Female speakers typically have higher formant frequencies due to shorter vocal tracts.
- **Fundamental Frequency (F0):** Female voices generally have higher pitch (F0) visible in the lower mel bins.
- **Spectral Tilt:** Gender differences affect the overall spectral energy distribution.
- **Temporal Patterns:** While the digit identity is preserved, individual speaking characteristics differ.

