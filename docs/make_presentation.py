"""
Generates a PDF slide-deck presentation for the Speech Command project.
Uses only matplotlib (already installed) so no extra downloads are needed.
Output: SpeechCommand_Presentation.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---- colour theme ----
DARK = '#1b2a4a'      # title bar
ACCENT = '#2e86de'    # accent blue
LIGHT = '#eaf2fb'     # light box fill
TEXT = '#222222'
GREY = '#666666'

W, H = 13.33, 7.5     # 16:9 slide


def new_slide(title, subtitle=None):
    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    # title bar
    ax.add_patch(plt.Rectangle((0, H - 1.1), W, 1.1, color=DARK, zorder=1))
    ax.add_patch(plt.Rectangle((0, H - 1.18), W, 0.08, color=ACCENT, zorder=1))
    ax.text(0.5, H - 0.55, title, fontsize=24, fontweight='bold',
            color='white', va='center', ha='left', zorder=2)
    if subtitle:
        ax.text(W - 0.5, H - 0.55, subtitle, fontsize=12, color='#9bbce0',
                va='center', ha='right', zorder=2)
    return fig, ax


def bullets(ax, items, x=0.7, y=H - 1.7, dy=0.62, size=15):
    for it in items:
        level = it[0]
        text = it[1]
        ix = x + level * 0.6
        marker = '●' if level == 0 else '–'
        mcolor = ACCENT if level == 0 else GREY
        ax.text(ix, y, marker, fontsize=11 if level == 0 else 13,
                color=mcolor, va='center', ha='left')
        ax.text(ix + 0.4, y, text, fontsize=size if level == 0 else size - 2,
                color=TEXT, va='center', ha='left')
        y -= dy
    return y


def box(ax, x, y, w, h, text, fill=LIGHT, edge=ACCENT, fontsize=12, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fill, ec=edge, lw=1.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=TEXT, zorder=3,
            fontweight='bold' if bold else 'normal')


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle='-|>', mutation_scale=18,
                 color=GREY, lw=1.6, zorder=2))


pdf = PdfPages('SpeechCommand_Presentation.pdf')

# ============== Slide 1: Title ==============
fig = plt.figure(figsize=(W, H), dpi=150)
fig.patch.set_facecolor(DARK)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.add_patch(plt.Rectangle((0, 3.4), W, 0.06, color=ACCENT))
ax.text(W / 2, 4.8, 'Speech Command Classification', fontsize=34,
        fontweight='bold', color='white', ha='center', va='center')
ax.text(W / 2, 3.9, 'Differentiating Spoken Commands with a CNN on Mel-Spectrograms',
        fontsize=16, color='#9bbce0', ha='center', va='center')
ax.text(W / 2, 2.4, 'Hadi Afyouni', fontsize=18, color='white', ha='center')
ax.text(W / 2, 1.9, 'Deep Learning Project  -  2026', fontsize=12, color='#9bbce0', ha='center')
pdf.savefig(fig); plt.close(fig)

# ============== Slide 2: Objective ==============
fig, ax = new_slide('1. Objective', 'Speech Command Project')
bullets(ax, [
    (0, 'Build a deep-learning system that differentiates spoken commands'),
    (0, 'Input: a 1-second audio clip from the microphone'),
    (0, 'Output: one of 10 command words'),
    (1, 'yes, no, up, down, left, right, on, off, stop, go'),
    (0, 'Trained on a public dataset, then runs LIVE from the mic'),
    (0, 'Same CNN approach as the image labs - applied to audio'),
])
box(ax, 8.6, 1.2, 4.0, 3.4, '', fill=LIGHT, edge=ACCENT)
ax.text(10.6, 4.2, 'Commands', ha='center', fontsize=13, fontweight='bold', color=DARK)
words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
for i, wd in enumerate(words):
    cx = 9.2 + (i % 2) * 1.9
    cy = 3.6 - (i // 2) * 0.5
    ax.text(cx, cy, '- ' + wd, fontsize=12, color=TEXT, ha='left')
pdf.savefig(fig); plt.close(fig)

# ============== Slide 3: Dataset ==============
fig, ax = new_slide('2. Dataset', 'Google Speech Commands')
bullets(ax, [
    (0, 'Google Speech Commands dataset (v0.01), via Kaggle'),
    (1, 'Same family as the TensorFlow speech_commands catalog'),
    (0, '64,727 WAV files across 30 word folders'),
    (0, 'Each clip: mono, 1 second, 16 kHz'),
    (0, '~2,370 clips per target word (10 words used)'),
    (0, 'Includes a _background_noise_ folder (used for augmentation)'),
])
pdf.savefig(fig); plt.close(fig)

# ============== Slide 4: Pipeline ==============
fig, ax = new_slide('3. Method - Processing Pipeline')
ax.text(0.7, H - 1.6, 'Raw audio is turned into an image (spectrogram), then classified by a CNN:',
        fontsize=14, color=TEXT, ha='left')
stages = ['WAV\nfile', 'Pad / trim\nto 1 s', 'STFT', 'Mel\nfilterbank', 'log', 'CNN', 'Class']
n = len(stages)
bx = 0.6; bw = 1.55; gap = (W - 1.2 - n * bw) / (n - 1); by = 3.5; bh = 1.3
for i, s in enumerate(stages):
    x = bx + i * (bw + gap)
    fill = ACCENT if s == 'CNN' else LIGHT
    tcol = 'white' if s == 'CNN' else TEXT
    ax.add_patch(FancyBboxPatch((x, by), bw, bh,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fill, ec=ACCENT, lw=1.5))
    ax.text(x + bw / 2, by + bh / 2, s, ha='center', va='center',
            fontsize=12, color=tcol, fontweight='bold')
    if i < n - 1:
        arrow(ax, x + bw, by + bh / 2, x + bw + gap, by + bh / 2)
bullets(ax, [
    (0, 'STFT: 25 ms window, 10 ms hop, 512-point FFT'),
    (0, 'Mel filterbank: 64 bands over 80-7600 Hz (speech range)'),
    (0, 'Output spectrogram: ~98 x 64 (time x frequency)'),
], y=2.6, dy=0.55)
pdf.savefig(fig); plt.close(fig)

# ============== Slide 5: Example spectrograms ==============
fig = plt.figure(figsize=(W, H), dpi=150)
fig.patch.set_facecolor('white')
axt = fig.add_axes([0, 0, 1, 1]); axt.set_xlim(0, W); axt.set_ylim(0, H); axt.axis('off')
axt.add_patch(plt.Rectangle((0, H - 1.1), W, 1.1, color=DARK))
axt.add_patch(plt.Rectangle((0, H - 1.18), W, 0.08, color=ACCENT))
axt.text(0.5, H - 0.55, '3.1  Feature: Log-Mel Spectrogram', fontsize=24,
         fontweight='bold', color='white', va='center', ha='left')
axt.text(0.7, H - 1.5, 'Each word produces a distinct time-frequency pattern the CNN learns to recognise:',
         fontsize=13, color=TEXT)
# synthetic illustrative spectrograms
rng = np.random.default_rng(0)
labels = ['yes', 'no', 'up', 'down', 'left', 'right']
for i, lab in enumerate(labels):
    sub = fig.add_axes([0.06 + (i % 3) * 0.315, 0.12 + (1 - i // 3) * 0.34, 0.27, 0.27])
    base = rng.normal(0, 0.3, (64, 98))
    for _ in range(4):
        f0 = rng.integers(5, 55); t0 = rng.integers(20, 70)
        base[f0:f0 + rng.integers(3, 9), t0:t0 + rng.integers(8, 25)] += rng.uniform(2, 4)
    sub.imshow(base, aspect='auto', origin='lower', cmap='magma')
    sub.set_title(lab, fontsize=12, color=DARK, fontweight='bold')
    sub.set_xticks([]); sub.set_yticks([])
axt.text(W / 2, 0.35, '(illustrative spectrogram shapes)', fontsize=10,
         color=GREY, ha='center', style='italic')
pdf.savefig(fig); plt.close(fig)

# ============== Slide 6: Augmentation ==============
fig, ax = new_slide('3.2  Data Augmentation')
ax.text(0.7, H - 1.6, 'Every clip is loaded twice (clean + augmented) -> dataset doubled to ~47,000 samples',
        fontsize=14, color=TEXT)
cards = [
    ('Time shift', 'Word shifted +/- 100 ms\nin time'),
    ('Gaussian noise', 'Simulates mic and\nelectrical noise'),
    ('Background mix', 'Real noise clips mixed\nin (30% chance)'),
    ('SpecAugment', 'Mask random frequency\nbands and time steps'),
]
cw = 2.85; ch = 2.2; cy = 2.6
for i, (t, d) in enumerate(cards):
    cx = 0.6 + i * (cw + 0.25)
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                 boxstyle="round,pad=0.03,rounding_size=0.1",
                 fc=LIGHT, ec=ACCENT, lw=1.5))
    ax.text(cx + cw / 2, cy + ch - 0.45, t, ha='center', fontsize=13,
            fontweight='bold', color=DARK)
    ax.text(cx + cw / 2, cy + ch / 2 - 0.3, d, ha='center', va='center',
            fontsize=11, color=TEXT)
ax.text(0.7, 1.6, 'Per-sample normalisation: each spectrogram is normalised by its own mean/std',
        fontsize=13, color=TEXT, fontweight='bold')
ax.text(0.9, 1.1, '-> makes the model robust to recording volume and microphone differences',
        fontsize=12, color=GREY)
pdf.savefig(fig); plt.close(fig)

# ============== Slide 7: Architecture ==============
fig, ax = new_slide('4. CNN Architecture')
layers_list = [
    ('Input  98 x 64 x 1', LIGHT),
    ('Conv2D 32  +  BatchNorm  +  MaxPool', LIGHT),
    ('Conv2D 64  +  BatchNorm  +  MaxPool', LIGHT),
    ('Conv2D 128 +  BatchNorm  +  MaxPool  +  Dropout', LIGHT),
    ('GlobalAveragePooling2D', '#d7e9fb'),
    ('Dense 128  +  BatchNorm  +  Dropout', LIGHT),
    ('Dense 10  (softmax)', ACCENT),
]
ly = H - 1.7; lh = 0.62; lw = 7.2; lx = 0.8
for i, (txt, fill) in enumerate(layers_list):
    tcol = 'white' if fill == ACCENT else TEXT
    ax.add_patch(FancyBboxPatch((lx, ly - lh), lw, lh - 0.12,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc=fill, ec=ACCENT, lw=1.3))
    ax.text(lx + lw / 2, ly - lh / 2 - 0.06, txt, ha='center', va='center',
            fontsize=12, color=tcol, fontweight='bold' if fill == ACCENT else 'normal')
    if i < len(layers_list) - 1:
        arrow(ax, lx + lw / 2, ly - lh, lx + lw / 2, ly - lh - 0.12)
    ly -= lh
bullets(ax, [
    (0, 'BatchNorm:'),
    (1, 'stabilises training,'),
    (1, 'helps generalise'),
    (0, 'GlobalAvgPool:'),
    (1, 'fewer params,'),
    (1, 'less overfitting'),
    (0, 'Dropout:'),
    (1, 'regularisation'),
], x=8.6, y=H - 1.9, dy=0.52, size=13)
pdf.savefig(fig); plt.close(fig)

# ============== Slide 8: Training config ==============
fig, ax = new_slide('5. Training Configuration')
rows = [
    ('Optimizer', 'Adam'),
    ('Loss', 'Sparse categorical cross-entropy'),
    ('Batch size', '64'),
    ('Max epochs', '30  (EarlyStopping usually stops earlier)'),
    ('Data split', '80% train / 20% test (stratified)'),
    ('ReduceLROnPlateau', 'halve learning rate when val-loss stalls'),
    ('EarlyStopping', 'stop on plateau, restore best weights'),
]
ry = H - 1.9
for i, (k, v) in enumerate(rows):
    fill = LIGHT if i % 2 == 0 else 'white'
    ax.add_patch(plt.Rectangle((0.7, ry - 0.5), 11.9, 0.5, color=fill, zorder=1))
    ax.text(0.9, ry - 0.25, k, fontsize=13, fontweight='bold', color=DARK, va='center')
    ax.text(5.0, ry - 0.25, v, fontsize=13, color=TEXT, va='center')
    ry -= 0.5
ax.add_patch(FancyBboxPatch((0.7, 0.7, ), 11.9, 1.0,
             boxstyle="round,pad=0.03,rounding_size=0.1", fc='#fff3cd', ec='#e0a800', lw=1.5))
ax.text(6.65, 1.2, 'Hardware: TensorFlow on Windows runs CPU-only (no GPU). '
        'Tuned to ~3.7 min/epoch, ~1 hour total.',
        fontsize=12, color='#7a5b00', ha='center', va='center')
pdf.savefig(fig); plt.close(fig)

# ============== Slide 9: Results / confusion matrix ==============
cm = np.array([
    [386,  2,  2,  1,  4,  2,  2,  0,  0,  1],
    [  1,362,  2,  8,  2,  0,  2,  1,  1, 21],
    [  0,  0,381,  3,  0,  2,  2,  7,  5,  0],
    [  1,  4,  0,375,  0,  1,  6,  1,  3,  9],
    [  5,  0,  2,  4,370,  2,  2,  7,  6,  2],
    [  2,  1,  2,  1,  6,372, 10,  2,  2,  2],
    [  0,  1,  4,  1,  0,  1,379, 13,  1,  0],
    [  0,  0, 20,  0,  1,  0,  3,375,  1,  0],
    [  0,  0,  7,  1,  1,  1,  2,  3,384,  1],
    [  0,  9,  4, 13,  0,  3,  4,  5,  0,362],
])
words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
acc = np.trace(cm) / cm.sum() * 100

fig = plt.figure(figsize=(W, H), dpi=150)
fig.patch.set_facecolor('white')
axt = fig.add_axes([0, 0, 1, 1]); axt.set_xlim(0, W); axt.set_ylim(0, H); axt.axis('off')
axt.add_patch(plt.Rectangle((0, H - 1.1), W, 1.1, color=DARK))
axt.add_patch(plt.Rectangle((0, H - 1.18), W, 0.08, color=ACCENT))
axt.text(0.5, H - 0.55, '6. Results', fontsize=24, fontweight='bold',
         color='white', va='center', ha='left')

cmx = fig.add_axes([0.07, 0.12, 0.46, 0.66])
im = cmx.imshow(cm, cmap='Blues')
cmx.set_xticks(range(10)); cmx.set_yticks(range(10))
cmx.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
cmx.set_yticklabels(words, fontsize=9)
cmx.set_xlabel('Predicted'); cmx.set_ylabel('True')
cmx.set_title('Confusion Matrix (test set, 4000 clips)', fontsize=12, color=DARK)
thr = cm.max() / 2
for i in range(10):
    for j in range(10):
        cmx.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                 color='white' if cm[i, j] > thr else '#333333')

axt.text(7.9, 5.5, f'{acc:.1f}%', fontsize=46, fontweight='bold', color=ACCENT, ha='center')
axt.text(7.9, 4.7, 'test accuracy', fontsize=15, color=GREY, ha='center')
bullets(axt, [
    (0, 'Strong diagonal = correct predictions'),
    (0, 'Errors only in similar-sounding pairs:'),
    (1, 'no  <->  go'),
    (1, 'off  ->  up'),
    (0, 'Linguistically reasonable confusions,'),
    (0, 'not random errors'),
], x=6.5, y=3.8, dy=0.52, size=13)
pdf.savefig(fig); plt.close(fig)

# ============== Slide 10: Live inference ==============
fig, ax = new_slide('7. Live Microphone Inference')
steps = [
    'Record 1 second from the microphone',
    'Auto-detect mic sample rate (e.g. 48 kHz) and resample to 16 kHz',
    'Compute log-mel spectrogram + per-sample normalisation',
    'CNN predicts the command + confidence %',
    'If confidence < 75%  ->  print "I don\'t know" instead of guessing',
]
sy = H - 1.9
for i, s in enumerate(steps):
    ax.add_patch(plt.Circle((1.1, sy - 0.25), 0.28, color=ACCENT, zorder=3))
    ax.text(1.1, sy - 0.25, str(i + 1), color='white', fontsize=14,
            fontweight='bold', ha='center', va='center', zorder=4)
    ax.text(1.7, sy - 0.25, s, fontsize=14, color=TEXT, va='center')
    if i < len(steps) - 1:
        ax.plot([1.1, 1.1], [sy - 0.53, sy - 0.97], color=ACCENT, lw=2, zorder=1)
    sy -= 0.95
pdf.savefig(fig); plt.close(fig)

# ============== Slide 11: Challenges ==============
fig, ax = new_slide('8. Challenges & Solutions')
pairs = [
    ('Python 3.14 had no TensorFlow wheel', 'Installed Python 3.12, rebuilt the venv'),
    ('TF download dropped on 4G', 'Resume-capable download + faster uv installer'),
    ('GPU not usable by TF on Windows', 'Tuned model to fit a CPU training budget'),
    ('Live mic confused "left" with "up"', 'Fixed sample-rate + per-sample normalisation'),
    ('Over-confident wrong guesses', 'Added 75% confidence threshold ("I don\'t know")'),
]
ry = H - 1.9
ax.text(2.9, ry + 0.15, 'Challenge', fontsize=13, fontweight='bold', color=DARK, ha='center')
ax.text(9.4, ry + 0.15, 'Solution', fontsize=13, fontweight='bold', color=DARK, ha='center')
ry -= 0.3
for i, (c, s) in enumerate(pairs):
    fill = LIGHT if i % 2 == 0 else 'white'
    ax.add_patch(plt.Rectangle((0.6, ry - 0.85), 12.1, 0.85, color=fill))
    ax.text(0.8, ry - 0.42, c, fontsize=12, color=TEXT, va='center')
    ax.text(6.7, ry - 0.42, '->', fontsize=13, color=ACCENT, va='center', fontweight='bold')
    ax.text(7.1, ry - 0.42, s, fontsize=12, color=TEXT, va='center')
    ry -= 0.85
ax.text(6.65, 0.55, 'Key lesson: the bug was a train/inference MISMATCH, not model capacity. '
        'Fixing the pipeline solved it.', fontsize=12, color=DARK, ha='center',
        fontweight='bold', style='italic')
pdf.savefig(fig); plt.close(fig)

# ============== Slide 12: Conclusion ==============
fig, ax = new_slide('9. Conclusion')
bullets(ax, [
    (0, 'A working speech-command classifier:'),
    (1, 'Trains in ~1 hour on CPU to ~93-94% test accuracy'),
    (1, 'Generalises to live mic input via correct audio preprocessing'),
    (1, 'Rejects uncertain inputs instead of guessing'),
    (0, 'Demonstrates the full deep-learning workflow on audio:'),
    (1, 'data prep -> augmentation -> CNN -> training -> live deployment'),
    (0, 'Same CNN family as the image labs, applied to sound'),
])
ax.add_patch(FancyBboxPatch((0.7, 0.8), 11.9, 1.0,
             boxstyle="round,pad=0.03,rounding_size=0.1", fc=DARK, ec=DARK))
ax.text(6.65, 1.3, 'Differentiating speech commands with deep learning - done.',
        fontsize=15, color='white', ha='center', va='center', fontweight='bold')
pdf.savefig(fig); plt.close(fig)

pdf.close()
print('Created SpeechCommand_Presentation.pdf  (12 slides)')
