# Speech Command Classification — Project Report

**Author:** Hadi Afyouni
**Date:** 2026-06-11
**File:** `LabML21_speechcmd_CNN.py`

---

## 1. Objective

Build a deep-learning system that **differentiates between spoken speech commands**.
The model listens to a 1-second audio clip and classifies it as one of **10 command words**:

```
yes, no, up, down, left, right, on, off, stop, go
```

The system is trained on a public dataset and then runs **live from the microphone**, so a
spoken word is recognised in real time.

---

## 2. Dataset

- **Source:** Google Speech Commands dataset (v0.01), downloaded from Kaggle.
- **Same family** as the official TensorFlow `speech_commands` catalog dataset (the catalog uses the newer v0.02).
- **Size:** 64,727 WAV files across 30 word folders + a `_background_noise_` folder.
- **Per class used:** ~2,370 clips for each of the 10 target words.
- **Format:** mono, 1 second, 16 kHz WAV files.

Only the 10 target words were loaded; the remaining words and the background-noise clips were
used for augmentation.

---

## 3. Method Overview

The pipeline converts raw audio into an **image-like representation** (a spectrogram) and then
classifies it with a **Convolutional Neural Network (CNN)** — the same family of models used for
image classification in the earlier labs (LabML17–LabML20).

```
WAV file  →  pad/truncate to 1 s  →  STFT  →  Mel filterbank  →  log  →  log-mel spectrogram  →  CNN  →  class
```

### 3.1 Feature extraction: log-mel spectrogram

Each clip is transformed into a 2-D time-frequency image:

| Parameter      | Value     | Meaning                              |
|----------------|-----------|--------------------------------------|
| Sample rate    | 16,000 Hz | Audio resolution                     |
| Window length  | 400 (25 ms) | STFT frame size                    |
| Hop length     | 160 (10 ms) | STFT step                          |
| FFT length     | 512       | Frequency resolution                 |
| Mel bins       | 64        | Number of mel frequency bands        |
| Frequency range| 80–7600 Hz| Covers the human speech range        |

Result: a spectrogram of shape **~98 × 64** (time × frequency), log-compressed so quiet and loud
parts are both visible.

### 3.2 Data augmentation

To make the model robust to different voices, microphones and noise, every clip is loaded twice —
once clean and once augmented — **doubling the dataset to ~47,000 samples**:

- **Random time shift** (±100 ms) — word can start at slightly different moments.
- **Gaussian noise** — simulates microphone / electrical noise.
- **Background-noise mixing** (30% chance) — real noise clips from the dataset's `_background_noise_` folder.
- **SpecAugment** — random masking of frequency bands and time steps directly on the spectrogram.

### 3.3 Normalisation

**Per-sample normalisation:** each spectrogram is normalised by *its own* mean and standard
deviation. This is critical — it makes the model insensitive to recording volume and microphone
differences, so a clip from the live microphone is treated the same way as a training clip.

---

## 4. CNN Architecture

A compact CNN sized to train within the time budget on CPU:

```
Input (98 × 64 × 1)
 → Conv2D(32) → BatchNorm → MaxPool
 → Conv2D(64) → BatchNorm → MaxPool
 → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3)
 → GlobalAveragePooling2D
 → Dense(128) → BatchNorm → Dropout(0.3)
 → Dense(10, softmax)
```

- **BatchNormalization** after each conv block stabilises training and improves generalisation.
- **GlobalAveragePooling** (instead of Flatten) reduces parameters and overfitting.
- **Dropout** further regularises the dense head.

---

## 5. Training Configuration

| Setting        | Value                                   |
|----------------|-----------------------------------------|
| Optimizer      | Adam                                    |
| Loss           | Sparse categorical cross-entropy        |
| Batch size     | 64                                      |
| Max epochs     | 30                                      |
| Split          | 80% train / 20% test (stratified)       |
| Callbacks      | ReduceLROnPlateau, EarlyStopping        |

- **ReduceLROnPlateau** — halves the learning rate when validation loss stops improving.
- **EarlyStopping** — stops training when validation loss plateaus and restores the best weights.

**Hardware reality:** TensorFlow 2.21 on native Windows runs **CPU-only** (GPU support was dropped
after TF 2.11), so the NVIDIA 3060 is not used. Training was tuned to fit a **~1–1.5 hour** total
budget on the CPU (~3.7 min/epoch).

---

## 6. Results

On the held-out 20% test set (4,000 clips), the model reached approximately **93.6% accuracy**.

The confusion matrix shows a strong diagonal. The few errors are concentrated in
**acoustically similar word pairs**, which is expected:

- `no` ↔ `go` (similar vowel + ending)
- `off` ↔ `up` (short vowel sounds)

These are linguistically reasonable confusions, not random errors.

---

## 7. Live Microphone Inference

After training, the model is saved (`speech_cmd_cnn.keras`) and reused without retraining.
The live loop:

1. **Records 1 second** from the default microphone.
2. **Auto-detects the microphone's sample rate** (the HyperX Cloud 2 routes through the Realtek
   codec at 48,000 Hz) and **resamples to 16,000 Hz** to match the training data.
3. Computes the log-mel spectrogram and applies the same per-sample normalisation.
4. Predicts the command and prints it with a confidence percentage.
5. **"I don't know" rejection:** if confidence is below 75%, the system declines to guess instead
   of outputting a wrong class.

---

## 8. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Python 3.14 had no TensorFlow wheel | Installed Python 3.12 alongside and rebuilt the venv |
| TF download kept dropping on 4G | Used resume-capable download; faster `uv` installer |
| GPU not usable by TF on Windows | Tuned model to fit a CPU training budget |
| Live mic confused `left` with `up` | **Root cause: train/inference mismatch** — fixed sample-rate (48k→16k) and switched to per-sample normalisation |
| Over-confident wrong guesses | Added a 75% confidence threshold ("I don't know") |

The key lesson: the misclassification was **not** a model-capacity problem. The original simple
model already reached ~94%. The live errors came from feeding the model audio that didn't match
the training distribution (wrong sample rate + global normalisation). Fixing the **pipeline**, not
enlarging the **network**, solved it.

---

## 9. Files

| File | Purpose |
|------|---------|
| `LabML21_speechcmd_CNN.py` | Main script (data prep, training, live demo) |
| `speech_commands/` | Dataset (10 word folders + background noise) |
| `speech_cmd_data.pkl` | Cached prepared spectrograms (skips slow reload) |
| `speech_cmd_cnn.keras` | Trained model (skips retraining) |

---

## 10. Conclusion

The project delivers a working speech-command classifier that:

- Trains in ~1 hour on CPU to ~93–94% test accuracy,
- Generalises to live microphone input through correct audio preprocessing,
- Rejects uncertain inputs instead of guessing.

It demonstrates the full deep-learning workflow — data preparation, augmentation, CNN design,
training with callbacks, evaluation, and real-time deployment — applied to audio rather than images.
