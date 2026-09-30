"""
Wake-word model: a small binary CNN answering "was that 'marvin'?" (yes/no).

Positives: every "marvin" clip in the dataset (clean + augmented copy).
Negatives, so it learns what "not marvin" sounds like:
  - "sheila" — a similar-sounding word, so it learns to tell look-alikes apart
  - an even sample of the other 28 words
  - background-noise crops
No extra download: it uses the same dataset as train_commands.py.

Run:  python train_wake_word.py      (~5-8 min on CPU)
  - saves models/wake_word_cnn.keras and training curves in docs/figures/
  - if the model already exists it is only evaluated — delete it to retrain.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                     GlobalAveragePooling2D, MaxPooling2D)

import config
from audio_features import (augment, compute_logmel, load_background_noise, load_wav,
                            normalize_sample, samples_per_clip)

wake_word = config.WAKE_WORD
hard_negative = config.WAKE_HARD_NEGATIVE
N_NOISE_NEGATIVES = 200
os.makedirs(config.FIGURES_DIR, exist_ok=True)

bg_clips = load_background_noise(config.DATASET_DIR)
print('Background noise clips loaded:', len(bg_clips))


def wav_files(word):
    folder = os.path.join(config.DATASET_DIR, word)
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]


# =========================
# Build the dataset (cached to a pickle)
# =========================
if not os.path.exists(config.WAKE_CACHE):
    X_list, y_list = [], []

    # --- positives: the wake word, clean + augmented copy ---
    files = wav_files(wake_word)
    print('Loading wake word:', wake_word, ' files:', len(files), '(x2 with augmentation)')
    for path in files:
        try:
            x = load_wav(path)
            X_list += [compute_logmel(x), compute_logmel(augment(x, bg_clips))]
            y_list += [1, 1]
        except Exception as e:
            print('skip', path, e)
    n_positive = len(y_list)

    # --- hard negative: a similar-sounding word, clean only ---
    files = wav_files(hard_negative)
    print('Loading hard negative:', hard_negative, ' files:', len(files))
    for path in files:
        try:
            X_list.append(compute_logmel(load_wav(path)))
            y_list.append(0)
        except Exception as e:
            print('skip', path, e)
    n_hard = len(files)

    # --- other words: sampled evenly so negatives roughly balance positives ---
    other_words = sorted(d for d in os.listdir(config.DATASET_DIR)
                         if os.path.isdir(os.path.join(config.DATASET_DIR, d))
                         and not d.startswith('_') and d not in (wake_word, hard_negative))
    budget = max(600, n_positive - n_hard - N_NOISE_NEGATIVES)
    per_word = max(1, budget // len(other_words))
    print('Sampling', per_word, 'clips each from', len(other_words), 'other words as negatives')
    for word in other_words:
        files = wav_files(word)
        for path in np.random.choice(files, size=min(per_word, len(files)), replace=False):
            try:
                X_list.append(compute_logmel(load_wav(path)))
                y_list.append(0)
            except Exception as e:
                print('skip', path, e)

    # --- pure background-noise crops ---
    if bg_clips:
        print('Sampling', N_NOISE_NEGATIVES, 'background-noise crops as negatives')
        for _ in range(N_NOISE_NEGATIVES):
            bg = bg_clips[np.random.randint(len(bg_clips))]
            start = np.random.randint(0, len(bg) - samples_per_clip)
            X_list.append(compute_logmel(bg[start:start + samples_per_clip]))
            y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(config.WAKE_CACHE, "wb") as f:
        pickle.dump({"X": X, "y": y, "wake_word": wake_word}, f)
else:
    with open(config.WAKE_CACHE, "rb") as f:
        var = pickle.load(f)
    X, y = var["X"], var["y"]
    print('Loaded cached wake-word dataset for:', var["wake_word"])
print('Positives:', int(y.sum()), ' Negatives:', int((y == 0).sum()))


# =========================
# Normalise, reshape, split
# =========================
for i in range(len(X)):
    X[i] = normalize_sample(X[i])
X = X[..., np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)
print('Train:', X_train.shape, ' Test:', X_test.shape)


# =========================
# Small binary CNN — lighter than the command model: "is it marvin?" is an easier question
# =========================
if os.path.exists(config.WAKE_MODEL):
    print('Loading saved model (delete it to retrain):', config.WAKE_MODEL)
    model = tf.keras.models.load_model(config.WAKE_MODEL)
else:
    model = models.Sequential([
        layers.Input(shape=X_train.shape[1:]),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),   # one number: probability it's the wake word
    ])
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6,
                                         restore_best_weights=True, verbose=1),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=config.WAKE_EPOCHS, batch_size=config.BATCH_SIZE,
                        callbacks=callbacks, verbose=1)

    metrics_df = pd.DataFrame(history.history)
    for cols, title, fname in [(["loss", "val_loss"], "Wake-word loss curve", 'wake_word_loss_curve.png'),
                               (["accuracy", "val_accuracy"], "Wake-word accuracy curve", 'wake_word_accuracy_curve.png')]:
        ax = metrics_df[cols].plot(title=title)
        ax.set_xlabel("Epoch")
        plt.savefig(os.path.join(config.FIGURES_DIR, fname))
        plt.show()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(config.WAKE_MODEL)
    print('Saved wake-word model to', config.WAKE_MODEL)


# =========================
# Evaluate on the 20% test clips
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
pred_labels = (model.predict(X_test, verbose=0)[:, 0] > 0.5).astype(int)
print('Confusion matrix [[TN, FP], [FN, TP]]:\n', confusion_matrix(y_test, pred_labels))
