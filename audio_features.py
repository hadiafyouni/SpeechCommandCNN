"""
Audio -> log-mel spectrogram, plus the data augmentation used in training.
Training and live inference both go through compute_logmel(), so the models
always see exactly the same kind of input they were trained on.
"""

import os

import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf

fs = 16000                 # all clips in this dataset are 16 kHz, 2x the highest speech frequency we keep
duration = 1               # 1 second per clip
samples_per_clip = fs * duration

frame_length = 400         # 25 ms * 16 samples/ms = 400 samples
frame_step = 160           # 10 ms * 16 samples/ms = 160 samples
fft_length = 512           # 400 samples + 112 zeros, a power of 2 so the FFT runs faster
num_mel_bins = 64          # compress the spectrum, keeping the frequencies that matter for speech
lower_hz = 80.0
upper_hz = 7600.0

_mel_matrix = None


def compute_logmel(x):
    """x: 1D float32 array, samples_per_clip long, range ~[-1, 1] -> (98, 64) log-mel image."""
    # STFT (Short-Time Fourier Transform): chop the audio into overlapping 25 ms
    # windows and ask "how much bass, mid and treble is in each one?". The result
    # is a 2D grid — X axis time, Y axis frequency (pitch), brightness = volume:
    # a picture of the sound.
    stft = tf.signal.stft(x, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectro = tf.abs(stft)

    # Linear -> mel: re-space the frequency axis the way human hearing does
    # (fine detail at low pitch, coarse at high pitch), 257 bins -> 64.
    global _mel_matrix
    if _mel_matrix is None:
        _mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins, num_spectrogram_bins=spectro.shape[-1],
            sample_rate=fs, lower_edge_hertz=lower_hz, upper_edge_hertz=upper_hz)
    mel_spectro = tf.matmul(spectro, _mel_matrix)

    # log compression: loudness is perceived logarithmically
    return tf.math.log(mel_spectro + 1e-6).numpy()


def fit_to_length(x):
    """Pad with zeros / truncate to exactly one second."""
    if len(x) < samples_per_clip:
        return np.pad(x, (0, samples_per_clip - len(x)))
    return x[:samples_per_clip]


def load_wav(file_path):
    """Read a 16-bit wav file -> float32 in [-1, 1], exactly one second long."""
    _, x = wav.read(file_path)
    x = x.astype(np.float32)
    if x.max() > 0:
        x = x / 32768.0
    return fit_to_length(x)


def normalize_sample(logmel):
    """Per-sample normalisation: each clip scaled by its own mean/std.
    Makes the model robust to volume and microphone differences."""
    m, s = logmel.mean(), logmel.std()
    return (logmel - m) / s if s > 0 else logmel


# =========================
# Augmentation (training only)
# =========================
def load_background_noise(dataset_dir):
    """The dataset's _background_noise_ recordings (running tap, fans, white noise...)."""
    clips = []
    noise_dir = os.path.join(dataset_dir, '_background_noise_')
    if os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.endswith('.wav'):
                try:
                    _, bg = wav.read(os.path.join(noise_dir, f))
                    bg = bg.astype(np.float32) / 32768.0
                    if len(bg) >= samples_per_clip:
                        clips.append(bg)
                except Exception:
                    pass
    return clips


def augment(x, bg_clips):
    """Instead of recording a million people saying "yes", take one recording
    and distort it into a new, slightly different variation."""
    # 1: random time shift up to 100 ms
    x = np.roll(x, np.random.randint(-1600, 1600))
    # 2: small gaussian noise (simulates mic noise / different recording conditions)
    x = x + np.random.normal(0, 0.005, x.shape).astype(np.float32)
    # 3: mix in a random background-noise clip at low volume (30% chance)
    if bg_clips and np.random.rand() < 0.3:
        bg = bg_clips[np.random.randint(len(bg_clips))]
        start = np.random.randint(0, len(bg) - samples_per_clip)
        x = x + np.random.uniform(0.02, 0.1) * bg[start:start + samples_per_clip]
    return np.clip(x, -1.0, 1.0)


def spec_augment(logmel):
    """SpecAugment: black out a random band of frequencies (horizontal stripe) and a
    random slice of time (vertical stripe), so the model can't rely on any one part."""
    logmel = logmel.copy()
    f = np.random.randint(0, 20)
    f0 = np.random.randint(0, max(1, num_mel_bins - f))
    logmel[:, f0:f0 + f] = logmel.min()
    t = np.random.randint(0, 20)
    t0 = np.random.randint(0, max(1, logmel.shape[0] - t))
    logmel[t0:t0 + t, :] = logmel.min()
    return logmel
