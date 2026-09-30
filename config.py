"""
Project-wide settings: file locations, vocabulary, training and live-listening
parameters. Every script reads from here, so a value only ever changes in one place.
"""

import os

# ---------- locations (relative to this file, so scripts run from any folder) ----------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, 'data', 'speech_commands')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
CACHE_DIR = os.path.join(PROJECT_DIR, 'cache')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'docs', 'figures')

COMMAND_MODEL = os.path.join(MODELS_DIR, 'command_cnn.keras')
WAKE_MODEL = os.path.join(MODELS_DIR, 'wake_word_cnn.keras')
COMMAND_CACHE = os.path.join(CACHE_DIR, 'command_features.pkl')
WAKE_CACHE = os.path.join(CACHE_DIR, 'wake_word_features.pkl')

# ---------- vocabulary ----------
COMMAND_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
# model output order — must match the order used when the model was trained
CLASS_NAMES = COMMAND_WORDS + ['unknown']
WAKE_WORD = 'marvin'
WAKE_HARD_NEGATIVE = 'sheila'   # similar-sounding word, trained on as a "not the wake word" example

# ---------- training ----------
COMMAND_EPOCHS = 30
WAKE_EPOCHS = 30
BATCH_SIZE = 64

# ---------- live listening ----------
COMMAND_CONF_THRESHOLD = 75.0    # %, below this the answer is "I don't know"
WAKE_THRESHOLD = 0.85            # wake-word probability needed to count as a hit
CONSECUTIVE_HOPS_REQUIRED = 2    # hits in a row needed to wake (filters one-off spikes)
HOP_SECONDS = 0.25               # how often the wake model re-checks the last second
BLOCK_SECONDS = 0.1              # size of each chunk the microphone driver delivers
# Loudest 100 ms of a window must exceed this RMS or the wake model isn't run:
# per-sample normalisation blows near-silence up into hiss the model never saw
# in training, and it guessed ~0.8 on it. On the dev laptop a quiet room peaked
# ~0.0005 and a spoken "marvin" measured ~0.0023.
SPEECH_RMS_GATE = 0.001
