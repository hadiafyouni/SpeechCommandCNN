"""
Push-to-talk GUI: click Record, say one command, see the prediction, its
confidence and the spectrogram the model looked at. No wake word needed.
Tkinter + embedded matplotlib -> nothing extra to install.

Run:  python push_to_talk.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # quiet TensorFlow startup logs

import threading
import tkinter as tk
from tkinter import font as tkfont

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import voice_engine as ve
from audio_features import fs

DARK = '#1b2a4a'
ACCENT = '#2e86de'
GREEN = '#27ae60'
ORANGE = '#e67e22'
LIGHTBG = '#f4f7fb'


def record_and_predict():
    """Record 1 s from the mic -> (log-mel spectrogram, predicted word, confidence %)."""
    recording = sd.rec(ve.window_len, samplerate=ve.MIC_RATE, channels=1, dtype='float32')
    sd.wait()
    x = recording.reshape(-1)
    word, conf = ve.classify(x)
    return ve.to_logmel(x), word, conf


class PushToTalkApp:
    def __init__(self, root):
        self.root = root
        root.title("Speech Command Recognizer")
        root.minsize(760, 0)   # height follows the content, so nothing is cut off at >100% display scaling
        root.configure(bg=LIGHTBG)

        self.f_title = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.f_word = tkfont.Font(family="Segoe UI", size=52, weight="bold")
        self.f_small = tkfont.Font(family="Segoe UI", size=11)
        self.f_btn = tkfont.Font(family="Segoe UI", size=16, weight="bold")

        header = tk.Frame(root, bg=DARK, height=70)
        header.pack(fill="x")
        tk.Label(header, text="Speech Command Recognizer", bg=DARK, fg="white",
                 font=self.f_title).pack(pady=16)

        self.word_var = tk.StringVar(value="--")
        self.word_lbl = tk.Label(root, textvariable=self.word_var, bg=LIGHTBG, fg=DARK, font=self.f_word)
        self.word_lbl.pack(pady=(22, 4))

        self.conf_var = tk.StringVar(value="press Record and say a command")
        tk.Label(root, textvariable=self.conf_var, bg=LIGHTBG, fg="#555", font=self.f_small).pack()

        self.bar = tk.Canvas(root, width=420, height=22, bg="#dfe6ee", highlightthickness=0)
        self.bar.pack(pady=10)
        self.bar_rect = self.bar.create_rectangle(0, 0, 0, 22, fill=ACCENT, width=0)

        self.fig, self.ax = plt.subplots(figsize=(5.6, 2.4), dpi=100)
        self.fig.patch.set_facecolor(LIGHTBG)
        self.ax.axis('off')
        self.ax.set_title("recorded audio (spectrogram)", fontsize=10, color="#555")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=6)
        self.canvas.draw()

        self.btn = tk.Button(root, text="🎤  Record  (1 s)", font=self.f_btn,
                             bg=ACCENT, fg="white", activebackground=DARK,
                             activeforeground="white", relief="flat",
                             padx=30, pady=12, command=self.on_record)
        self.btn.pack(pady=14)

        tk.Label(root, text="Commands:  " + "   ".join(config.COMMAND_WORDS),
                 bg=LIGHTBG, fg="#777", font=self.f_small).pack(pady=(4, 0))

        tk.Label(root, text=f"Mic: {ve.MIC_NAME}  @ {ve.MIC_RATE} Hz   |   resampled to {fs} Hz",
                 bg="#e4ebf3", fg="#555", font=("Segoe UI", 9), anchor="w").pack(side="bottom", fill="x")

    def set_bar(self, conf, color):
        self.bar.coords(self.bar_rect, 0, 0, int(420 * min(conf, 100) / 100), 22)
        self.bar.itemconfig(self.bar_rect, fill=color)

    def on_record(self):
        self.btn.config(state="disabled", text="● Recording...")
        self.word_var.set("...")
        self.conf_var.set("listening")
        self.set_bar(0, ACCENT)
        self.root.update_idletasks()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        result = record_and_predict()
        self.root.after(0, self._show_result, *result)   # Tk must only be touched from its own thread

    def _show_result(self, logmel, word, conf):
        if word == 'unknown' or conf < config.COMMAND_CONF_THRESHOLD:
            self.word_var.set("?")
            self.word_lbl.config(fg=ORANGE)
            self.conf_var.set(ve.describe(word, conf))
            self.set_bar(conf, ORANGE)
        else:
            self.word_var.set(word.upper())
            self.word_lbl.config(fg=GREEN)
            self.conf_var.set(f"confidence: {conf:.1f}%")
            self.set_bar(conf, GREEN)

        self.ax.clear()
        self.ax.imshow(logmel.T, aspect='auto', origin='lower', cmap='magma')
        self.ax.axis('off')
        self.ax.set_title("recorded audio (spectrogram)", fontsize=10, color="#555")
        self.canvas.draw()

        self.btn.config(state="normal", text="🎤  Record  (1 s)")


if __name__ == "__main__":
    root = tk.Tk()
    PushToTalkApp(root)
    root.mainloop()
