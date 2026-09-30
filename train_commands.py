"""
Speech command classification: CNN on log-mel spectrograms.
10 command words + an "unknown" class (any other word, noise or silence).
(originally LabML21_speechcmd_CNN.py)

Dataset: Google Speech Commands (Kaggle, ~1 GB), unzipped so the word folders
sit directly in data/speech_commands/ (yes/, no/, ..., _background_noise_/).
  https://www.kaggle.com/datasets/neehakurelli/google-speech-commands

Run:  python train_commands.py
  - first run: builds + caches features (~15 min), trains (~1.5 h on CPU),
    saves models/command_cnn.keras and training curves in docs/figures/
  - if models/command_cnn.keras already exists: skips training and only
    evaluates it (accuracy + confusion matrix). Delete the model to retrain.
"""

import os
import pickle
from datetime import date

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
from audio_features import (augment, compute_logmel, fft_length, frame_length, frame_step,
                            load_background_noise, load_wav, normalize_sample, num_mel_bins,
                            samples_per_clip, spec_augment)

# =========================
# Part 0: Parameters
# =========================
command_words = config.COMMAND_WORDS
class_names = config.CLASS_NAMES
Nbclasses = len(class_names)
os.makedirs(config.FIGURES_DIR, exist_ok=True)

# background noise clips for augmentation, so the model learns to ignore them
bg_clips = load_background_noise(config.DATASET_DIR)
print('Background noise clips loaded:', len(bg_clips))


# =========================
# Part 1: One clip -> log-mel spectrogram (optionally augmented)
# =========================
def clip_to_logmel(x, augment_audio=False):
    if not augment_audio:
        return compute_logmel(x)
    # waveform augmentation (shift, noise, background) then SpecAugment masks
    return spec_augment(compute_logmel(augment(x, bg_clips)))


# =========================
# Part 2: Build X, y from the dataset folders (cached to a pickle)
# Each file is used twice: once clean, once augmented -> doubles the dataset
# =========================
def build_unknown_class(target_size):
    """'unknown' examples: the other 20 words in the dataset + background noise.
    Sized like one command class so the model isn't biased toward or against it."""
    label = class_names.index('unknown')
    other_words = sorted(d for d in os.listdir(config.DATASET_DIR)
                         if os.path.isdir(os.path.join(config.DATASET_DIR, d))
                         and not d.startswith('_') and d not in command_words)
    n_noise = 300 if bg_clips else 0
    per_word = max(1, (target_size - n_noise) // (2 * len(other_words)))
    print('Loading class: unknown  from', len(other_words), 'other words,', per_word,
          'files each (x2 with augmentation) +', n_noise, 'noise crops')
    Xu, yu = [], []
    for word in other_words:
        folder = os.path.join(config.DATASET_DIR, word)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        for fname in np.random.choice(files, size=min(per_word, len(files)), replace=False):
            try:
                x = load_wav(os.path.join(folder, fname))
                Xu += [clip_to_logmel(x), clip_to_logmel(x, augment_audio=True)]
                yu += [label, label]
            except Exception as e:
                print('skip', fname, e)
    # noise with no word in it is also an "I don't know"
    for _ in range(n_noise):
        bg = bg_clips[np.random.randint(len(bg_clips))]
        start = np.random.randint(0, len(bg) - samples_per_clip)
        Xu.append(clip_to_logmel(bg[start:start + samples_per_clip]))
        yu.append(label)
    return Xu, yu


if not os.path.exists(config.COMMAND_CACHE):
    X_list = []
    y_list = []
    for label, word in enumerate(command_words):
        folder = os.path.join(config.DATASET_DIR, word)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print('Loading class:', word, '  number of files:', len(files), '(x2 with augmentation)')
        for fname in files:
            try:
                x = load_wav(os.path.join(folder, fname))
                X_list.append(clip_to_logmel(x))                      # a clean image of the word
                y_list.append(label)
                X_list.append(clip_to_logmel(x, augment_audio=True))  # a noisy, distorted, masked copy
                y_list.append(label)
            except Exception as e:
                print('skip', fname, e)

    Xu, yu = build_unknown_class(len(y_list) // len(command_words))
    X_list += Xu
    y_list += yu

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    var = {"X": X,
           "y": y,
           "Class_words": class_names,
           "Students": 'ESIGELEC Univ',
           "date": date.today(),
           "param": ['log-mel-spectrogram', frame_length, frame_step, fft_length, num_mel_bins]
           }
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(config.COMMAND_CACHE, "wb") as f:
        pickle.dump(var, f)
else:
    with open(config.COMMAND_CACHE, "rb") as f:
        var = pickle.load(f)
    X = var["X"]
    y = var["y"]
    print(var["Class_words"])
    print(var["Students"])
    print(var["date"])


# =========================
# Part 3: Visualise one spectrogram per class
# =========================
plt.figure(figsize=(12, 6))
for i, word in enumerate(class_names):
    idx = np.where(y == i)[0][0]
    plt.subplot(2, 6, i + 1)
    plt.imshow(X[idx].T, aspect='auto', origin='lower', cmap='magma')
    plt.title(word)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'spectrogram_per_class.png'))
plt.show()


# =========================
# Part 4: Normalise + reshape for CNN
# =========================
for i in range(len(X)):
    X[i] = normalize_sample(X[i])

# Add channel dim --> (samples, time, mel, 1)
# CNNs expect a 3D input (height, width, channels) for each sample.
# Our log-mel spectrograms are 2D (time x mel); adding an axis at the end makes
# a single-channel image (like a grayscale picture), which Conv2D expects.
X = X[..., np.newaxis]
print('X final shape:', X.shape)


# =========================
# Part 5: Train / test split (80% / 20%)
# fixed random_state -> the same split every run, so evaluating a saved model
# later still uses clips it never trained on
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)
print('Train:', X_train.shape, ' Test:', X_test.shape)


# =========================
# Part 6: CNN model — load if saved, else build + train
# =========================
if os.path.exists(config.COMMAND_MODEL):
    print('Loading saved model (delete it to retrain):', config.COMMAND_MODEL)
    model = tf.keras.models.load_model(config.COMMAND_MODEL)
    model.summary()
else:
    model = models.Sequential()
    model.add(layers.Input(shape=X_train.shape[1:]))

    # Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))  # by forcing the network to work with "brain damage",
                             # it can't memorise the specific voices in the training
                             # data and has to learn the general rules of the words

    # GlobalAveragePooling instead of Flatten — fewer params, less overfit, faster
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(Nbclasses, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # =========================
    # Part 7: Train
    # =========================
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6,
                                         restore_best_weights=True, verbose=1),
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=config.COMMAND_EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        callbacks=callbacks,
                        verbose=1)

    # =========================
    # Part 8: Training curves
    # =========================
    metrics_df = pd.DataFrame(history.history)
    for cols, title, fname in [(["loss", "val_loss"], "Loss curve", 'command_loss_curve.png'),
                               (["accuracy", "val_accuracy"], "Accuracy curve", 'command_accuracy_curve.png')]:
        ax = metrics_df[cols].plot(title=title)
        ax.set_xlabel("Epoch")
        plt.savefig(os.path.join(config.FIGURES_DIR, fname))
        plt.show()

    # =========================
    # Part 9: Save model
    # =========================
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(config.COMMAND_MODEL)
    print('Saved model to', config.COMMAND_MODEL)


# =========================
# Part 10: Evaluate + confusion matrix (on the 20% test clips)
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

pred_labels = model.predict(X_test, verbose=0).argmax(axis=1)
print('Confusion matrix (rows = true, columns = predicted):', class_names)
print(confusion_matrix(y_test, pred_labels))
print('\nTry it live:  python push_to_talk.py   |   python listener.py   |   python marvin_game.py')
