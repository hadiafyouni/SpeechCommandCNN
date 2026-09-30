"""
Live-audio engine shared by listener.py, marvin_game.py and push_to_talk.py:
loads both trained models, runs one microphone stream feeding a queue, and
turns raw microphone audio into predictions.

Two hard-won rules for the microphone on Windows (Realtek mic array):
  - Open ONE stream and keep it open. Stopping/reopening it between commands
    caused driver overflows and freezes.
  - Never play a sound while it is open. winsound.Beep or sounddevice output
    killed the input stream within one or two plays — which is why nothing
    in this project beeps.
"""

import queue

import numpy as np
import sounddevice as sd
import tensorflow as tf
from scipy.signal import resample_poly

import config
from audio_features import compute_logmel, duration, fit_to_length, fs, normalize_sample

print('Loading models...')
wake_model = tf.keras.models.load_model(config.WAKE_MODEL)
command_model = tf.keras.models.load_model(config.COMMAND_MODEL)

_mic = sd.query_devices(kind='input')
MIC_RATE = int(_mic['default_samplerate'])   # record at the mic's native rate, resample to 16 kHz
MIC_NAME = _mic['name']
window_len = int(MIC_RATE * duration)        # samples in one second of mic audio
hop_len = int(MIC_RATE * config.HOP_SECONDS)

# The callback runs on the audio driver's real-time thread: if it is slow the
# driver overflows, so it only hands chunks over to the main thread.
audio_q = queue.Queue()
overflow_count = 0


def audio_callback(indata, frames, time_info, status):
    global overflow_count
    if status.input_overflow:
        overflow_count += 1
    audio_q.put(indata[:, 0].copy())


def open_mic_stream():
    """Use as `with open_mic_stream():` — chunks then arrive in audio_q."""
    return sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='float32',
                          blocksize=int(MIC_RATE * config.BLOCK_SECONDS), latency='high',
                          callback=audio_callback)


def flush_queue():
    while True:
        try:
            audio_q.get_nowait()
        except queue.Empty:
            return


def next_samples(n):
    """Block until n fresh samples have arrived from the mic."""
    chunks, got = [], 0
    while got < n:
        chunk = audio_q.get()
        chunks.append(chunk)
        got += len(chunk)
    return np.concatenate(chunks)[:n]


def loudest_rms(x):
    """Loudness (RMS) of the loudest 100 ms in x."""
    blk = int(MIC_RATE * 0.1)
    frames = x[:len(x) // blk * blk].reshape(-1, blk)
    return float(np.sqrt((frames ** 2).mean(axis=1)).max())


def to_logmel(raw):
    """1 s of mic audio at MIC_RATE -> log-mel spectrogram, same pipeline as training."""
    x = raw
    if MIC_RATE != fs:
        x = resample_poly(x, up=fs, down=MIC_RATE).astype(np.float32)
    return compute_logmel(fit_to_length(x))


def to_model_input(raw):
    return normalize_sample(to_logmel(raw))[np.newaxis, ..., np.newaxis]


# model(x) instead of model.predict(x): predict() has heavy per-call overhead
# and got slower and slower when called several times a second.
def wake_probability(window):
    """-> (loudness, probability the last second contains the wake word)."""
    level = loudest_rms(window)
    if level < config.SPEECH_RMS_GATE:
        return level, 0.0   # nothing loud enough to be speech — don't ask the model
    return level, float(wake_model(to_model_input(window), training=False).numpy()[0][0])


def classify(clip):
    """-> (predicted class name, confidence in %)."""
    pred = command_model(to_model_input(clip), training=False).numpy()[0]
    idx = int(np.argmax(pred))
    return config.CLASS_NAMES[idx], float(pred[idx]) * 100


def describe(word, conf):
    """Text answer for a prediction, including the two kinds of "I don't know"."""
    if word == 'unknown':
        return f"I don't know (not one of my commands, {conf:.1f}%)"
    if conf < config.COMMAND_CONF_THRESHOLD:
        return f"I don't know (best guess: {word}, {conf:.1f}%)"
    return f'{word} ({conf:.1f}%)'


def warm_up():
    """First call to a Keras model is slow (graph setup) — do it before listening."""
    silent = to_model_input(np.zeros(window_len, dtype=np.float32))
    wake_model(silent, training=False)
    command_model(silent, training=False)
