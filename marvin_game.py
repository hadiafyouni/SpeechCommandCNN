"""
Marvin — a little robot you drive with your voice.

Say "marvin" to wake him up, then give commands:
  up / down / left / right   step one square that way
  go / stop                  keep walking / stop walking
  on / off                   lights on (day) / lights off (night)
  yes / no                   happy hop / head shake
He stays awake for AWAKE_SECONDS after the last command, then dozes off.
Walk him over the stars to collect them.

Keyboard works too (a backup if the mic misbehaves during a demo):
  M wake · arrows move · G go · S stop · O on · F off · Y yes · N no

Run:  python marvin_game.py
"""

import math
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import numpy as np

import config
import voice_engine as ve   # loads both models + the mic helpers

AWAKE_SECONDS = 10
ONSET_RMS = 0.001         # while awake, a 100 ms chunk louder than this starts a command capture
PRE_ROLL_SECONDS = 0.3    # audio kept from just before the onset, so the word's start isn't cut
IGNORE_AFTER_SECONDS = 0.4  # ignore sound right after waking/a command (tail of the last word)

GRID_W, GRID_H, CELL = 10, 7, 64
W, H = GRID_W * CELL, GRID_H * CELL
FRAME_MS = 16
WALK_PX = 4               # pixels per frame -> ~0.25 s per square

DIRS = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
PUPIL_SHIFT = {'up': (0, -2), 'down': (0, 2), 'left': (-3, 0), 'right': (3, 0)}
KEYS = {'Up': 'up', 'Down': 'down', 'Left': 'left', 'Right': 'right',
        'g': 'go', 's': 'stop', 'o': 'on', 'f': 'off', 'y': 'yes', 'n': 'no'}

DARK = '#1b2a4a'
GOLD = '#f5b700'
DAY_BG, DAY_GRID, DAY_SHADOW = '#e6f4ff', '#cfe6f7', '#b9d3e6'
NIGHT_BG, NIGHT_GRID, NIGHT_SHADOW = '#0e1630', '#1a2748', '#070c1c'
BODY, BODY_EDGE, VISOR = '#3fb6a8', '#257a71', '#15222e'
EYE, EYE_SLEEP = '#7df9ff', '#4d7479'


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{int(x + (y - x) * t):02x}' for x, y in zip(a, b))


def star_points(cx, cy, r_out, r_in):
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / 5
        pts += [cx + r * math.cos(a), cy + r * math.sin(a)]
    return pts


def approach(a, b, step):
    return b if abs(b - a) <= step else a + step * (1 if b > a else -1)


# =========================
# Audio thread: same wake-word logic as listener.py while asleep; while
# awake, every burst of sound is captured as a 1 s clip and classified.
# Results go to the UI through `events` (Tkinter isn't thread-safe).
# =========================
def audio_worker(events, stop_flag):
    try:
        _listen(events, stop_flag)
    except Exception as e:
        events.put(('error', f'mic error: {e} (keyboard still works)'))


def _listen(events, stop_flag):
    ve.warm_up()
    window_len, hop_len = ve.window_len, ve.hop_len
    pre_len = int(ve.MIC_RATE * PRE_ROLL_SECONDS)
    window = np.zeros(window_len, dtype=np.float32)
    fresh = since_check = streak = 0
    awake_until = None
    ignore_until = 0.0

    with ve.open_mic_stream():
        while not stop_flag.is_set():
            try:
                chunk = ve.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            window = np.concatenate([window[len(chunk):], chunk])
            fresh += len(chunk)
            since_check += len(chunk)
            now = time.time()

            if awake_until is not None:
                if now > awake_until:
                    awake_until = None
                    fresh = since_check = streak = 0
                    events.put(('sleep',))
                    continue
                if now < ignore_until or np.sqrt(np.mean(chunk ** 2)) < ONSET_RMS:
                    continue
                clip = np.concatenate([window[-pre_len:], ve.next_samples(window_len - pre_len)])
                events.put(('command', *ve.classify(clip)))
                ve.flush_queue()
                awake_until = time.time() + AWAKE_SECONDS
                ignore_until = time.time() + IGNORE_AFTER_SECONDS
                continue

            if fresh < window_len or since_check < hop_len:
                continue
            since_check = 0
            level, prob = ve.wake_probability(window)
            streak = streak + 1 if prob > config.WAKE_THRESHOLD else 0
            events.put(('status', level, prob))
            if streak >= config.CONSECUTIVE_HOPS_REQUIRED:
                streak = 0
                ve.flush_queue()
                awake_until = time.time() + AWAKE_SECONDS
                ignore_until = time.time() + IGNORE_AFTER_SECONDS
                events.put(('wake',))


# =========================
# The game window
# =========================
class MarvinGame:
    def __init__(self, root):
        self.root = root
        root.title('Marvin - voice-controlled robot')
        root.configure(bg=DARK)
        root.resizable(False, False)

        f_big = tkfont.Font(family='Segoe UI', size=13, weight='bold')
        f_small = tkfont.Font(family='Segoe UI', size=10)
        self.f_bubble = tkfont.Font(family='Segoe UI', size=12, weight='bold')

        top = tk.Frame(root, bg=DARK)
        top.pack(fill='x', padx=12, pady=(10, 4))
        self.state_lbl = tk.Label(top, font=f_big, bg=DARK, fg='white')
        self.state_lbl.pack(side='left')
        self.score_lbl = tk.Label(top, text='stars: 0', font=f_big, bg=DARK, fg=GOLD)
        self.score_lbl.pack(side='right')

        info = tk.Frame(root, bg=DARK)
        info.pack(fill='x', padx=12)
        self.heard_lbl = tk.Label(info, text='heard: -', font=f_small, bg=DARK, fg='#bcd')
        self.heard_lbl.pack(side='left')
        self.mic_lbl = tk.Label(info, text='mic starting...', font=f_small, bg=DARK, fg='#789')
        self.mic_lbl.pack(side='right')

        self.timer = tk.Canvas(root, width=W, height=6, bg=DARK, highlightthickness=0)
        self.timer.pack(pady=(4, 0))
        self.canvas = tk.Canvas(root, width=W, height=H, highlightthickness=0)
        self.canvas.pack()
        tk.Label(root, font=f_small, bg=DARK, fg='#9ab', justify='center',
                 text='Say "marvin" to wake him, then:  up · down · left · right · go · stop · on · off · yes · no\n'
                      'Keyboard: M wake · arrows move · G go · S stop · O on · F off · Y yes · N no'
                 ).pack(pady=8)

        self.cx, self.cy = GRID_W // 2, GRID_H // 2
        self.px, self.py = self.cx * CELL, self.cy * CELL
        self.facing = 'down'
        self.walking = False
        self.awake = False
        self.awake_until = 0.0
        self.lights = True
        self.anim, self.anim_t0 = None, 0.0
        self.bubble, self.bubble_until = None, 0.0
        self.score = 0
        self.star = self.random_free_cell()
        self.popups = []
        self.frame = 0
        self.sky = [(random.uniform(0, W), random.uniform(0, H), random.uniform(0, 6)) for _ in range(40)]

        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        root.bind('<Key>', self.on_key)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.set_asleep()
        self.tick()
        self.poll_events()

    # ---------- input ----------
    def poll_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == 'wake':
                    self.wake()
                elif ev[0] == 'sleep':
                    self.set_asleep()
                elif ev[0] == 'command':
                    self.command(ev[1], ev[2])
                elif ev[0] == 'status':
                    self.mic_lbl.config(text=f'mic level {ev[1]:.4f} · wake {ev[2]:.2f}')
                elif ev[0] == 'error':
                    self.mic_lbl.config(text=ev[1], fg='#ff8a8a')
        except queue.Empty:
            pass
        self.root.after(30, self.poll_events)

    def on_key(self, e):
        key = e.keysym if e.keysym in KEYS else e.keysym.lower()
        if key == 'm':
            self.wake()
        elif key in KEYS:
            self.command(KEYS[key], 100.0)

    def close(self):
        self.stop_flag.set()
        self.root.destroy()

    # ---------- behaviour ----------
    def set_awake(self):
        self.awake = True
        self.awake_until = time.time() + AWAKE_SECONDS
        self.state_lbl.config(text='Marvin is awake - give a command')

    def wake(self):
        self.set_awake()
        self.say("I'm up!", 1.2)

    def set_asleep(self):
        self.awake = False
        self.walking = False
        self.bubble = None
        self.state_lbl.config(text='Marvin is sleeping - say "marvin" to wake him')

    def command(self, word, conf):
        if not self.awake:
            self.set_awake()
        self.awake_until = time.time() + AWAKE_SECONDS
        if word == 'unknown':
            self.heard_lbl.config(text=f'heard: not one of my commands  ({conf:.0f}%)')
            self.say('?', 1.2)
            return
        if conf < config.COMMAND_CONF_THRESHOLD:
            self.heard_lbl.config(text=f'heard: ?  (best guess "{word}", only {conf:.0f}%)')
            self.say('?', 1.2)
            return
        self.heard_lbl.config(text=f'heard: "{word}"  ({conf:.0f}%)')
        if word in DIRS:
            self.facing = word
            self.walking = False
            self.say(word + '!', 0.8)
            self.step()
        elif word == 'go':
            self.walking = True
            self.say('go!', 0.8)
            self.step()
        elif word == 'stop':
            self.walking = False
            self.say('ok', 0.8)
        elif word == 'on':
            self.lights = True
            self.say('lights on', 1.0)
        elif word == 'off':
            self.lights = False
            self.say('lights off', 1.0)
        elif word == 'yes':
            self.start_anim('yes')
            self.say('yes!', 1.0)
        elif word == 'no':
            self.start_anim('no')
            self.say('no!', 1.0)

    def step(self):
        dx, dy = DIRS[self.facing]
        nx, ny = self.cx + dx, self.cy + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            self.cx, self.cy = nx, ny
        else:
            self.walking = False
            self.start_anim('bump')

    def arrived(self):
        if (self.cx, self.cy) == self.star:
            self.score += 1
            self.score_lbl.config(text=f'stars: {self.score}')
            self.popups.append((self.px + CELL / 2, self.py, time.time()))
            self.star = self.random_free_cell()
            self.start_anim('yes')
        if self.walking:
            self.step()

    def random_free_cell(self):
        while True:
            cell = (random.randrange(GRID_W), random.randrange(GRID_H))
            if cell != (self.cx, self.cy):
                return cell

    def say(self, text, seconds):
        self.bubble, self.bubble_until = text, time.time() + seconds

    def start_anim(self, name):
        self.anim, self.anim_t0 = name, time.time()

    def anim_offset(self, now):
        t = now - self.anim_t0
        if self.anim == 'yes' and t < 0.8:
            return 0, -abs(math.sin(t * math.pi / 0.4)) * 14
        if self.anim == 'no' and t < 0.6:
            return math.sin(t * math.pi * 2 / 0.15) * 5, 0
        if self.anim == 'bump' and t < 0.3:
            dx, dy = DIRS[self.facing]
            k = math.sin(t * math.pi / 0.3) * 6
            return dx * k, dy * k
        self.anim = None
        return 0, 0

    # ---------- frame loop ----------
    def tick(self):
        now = time.time()
        self.frame += 1
        if self.awake and now > self.awake_until + 1.0:
            self.set_asleep()
        self.move()
        self.draw(now)
        self.root.after(FRAME_MS, self.tick)

    def move(self):
        tx, ty = self.cx * CELL, self.cy * CELL
        if (self.px, self.py) == (tx, ty):
            return
        self.px = approach(self.px, tx, WALK_PX)
        self.py = approach(self.py, ty, WALK_PX)
        if (self.px, self.py) == (tx, ty):
            self.arrived()

    def draw(self, now):
        c = self.canvas
        c.delete('all')
        bg, grid = (DAY_BG, DAY_GRID) if self.lights else (NIGHT_BG, NIGHT_GRID)
        c.create_rectangle(0, 0, W, H, fill=bg, width=0)
        for i in range(1, GRID_W):
            c.create_line(i * CELL, 0, i * CELL, H, fill=grid)
        for j in range(1, GRID_H):
            c.create_line(0, j * CELL, W, j * CELL, fill=grid)
        if not self.lights:
            for sx, sy, phase in self.sky:
                col = mix(NIGHT_BG, '#e8ecff', 0.35 + 0.35 * math.sin(now * 2 + phase))
                c.create_oval(sx - 1.5, sy - 1.5, sx + 1.5, sy + 1.5, fill=col, width=0)

        sx, sy = self.star
        pulse = 1 + 0.08 * math.sin(now * 4)
        c.create_polygon(star_points(sx * CELL + CELL / 2, sy * CELL + CELL / 2, 17 * pulse, 7 * pulse),
                         fill=GOLD, outline='#c98f00', width=2)

        ox, oy = self.anim_offset(now)
        x, y = self.px + CELL / 2 + ox, self.py + CELL / 2 + oy
        self.draw_robot(x, y, now)

        alive = []
        for px, py, t0 in self.popups:
            t = now - t0
            if t < 0.9:
                c.create_text(px, py - t * 30, text='+1', font=self.f_bubble,
                              fill=mix(GOLD, bg, t / 0.9))
                alive.append((px, py, t0))
        self.popups = alive

        if self.bubble and now < self.bubble_until:
            tw = self.f_bubble.measure(self.bubble)
            bx = min(max(x, tw / 2 + 12), W - tw / 2 - 12)
            by = y - 58 if y - 58 > 14 else y + 52
            c.create_rectangle(bx - tw / 2 - 8, by - 12, bx + tw / 2 + 8, by + 12,
                               fill='white', outline=DARK, width=2)
            c.create_text(bx, by, text=self.bubble, font=self.f_bubble, fill=DARK)

        self.timer.delete('all')
        if self.awake:
            frac = max(0.0, (self.awake_until - now) / AWAKE_SECONDS)
            self.timer.create_rectangle(0, 0, W * frac, 6, fill='#27ae60', width=0)

    def draw_robot(self, x, y, now):
        c = self.canvas
        moving = (self.px, self.py) != (self.cx * CELL, self.cy * CELL)
        if self.awake and not moving:
            y += math.sin(now * 3) * 1.5

        c.create_oval(x - 18, y + 21, x + 18, y + 28,
                      fill=DAY_SHADOW if self.lights else NIGHT_SHADOW, width=0)

        phase = (self.frame // 6) % 2 if moving else -1
        lf, rf = (3 if phase == 0 else 0), (3 if phase == 1 else 0)
        c.create_rectangle(x - 13, y + 14 - lf, x - 4, y + 24 - lf, fill=BODY_EDGE, width=0)
        c.create_rectangle(x + 4, y + 14 - rf, x + 13, y + 24 - rf, fill=BODY_EDGE, width=0)
        c.create_oval(x - 29, y - 2, x - 19, y + 9, fill=BODY, outline=BODY_EDGE, width=2)
        c.create_oval(x + 19, y - 2, x + 29, y + 9, fill=BODY, outline=BODY_EDGE, width=2)

        c.create_line(x, y - 20, x, y - 31, fill=BODY_EDGE, width=3)
        if self.awake:
            g = int(90 + 70 * (0.5 + 0.5 * math.sin(now * 6)))
            bulb = f'#ff{g:02x}{g:02x}'
        else:
            bulb = '#8a8f99'
        c.create_oval(x - 5, y - 38, x + 5, y - 28, fill=bulb, outline=BODY_EDGE, width=2)

        c.create_oval(x - 23, y - 22, x + 23, y + 20, fill=BODY, outline=BODY_EDGE, width=3)
        c.create_rectangle(x - 15, y - 13, x + 15, y + 7, fill=VISOR, outline=BODY_EDGE, width=2)

        if self.awake:
            fx, fy = PUPIL_SHIFT[self.facing]
            for ex in (-7, 7):
                c.create_oval(x + ex + fx - 3.5, y - 6 + fy - 3.5, x + ex + fx + 3.5, y - 6 + fy + 3.5,
                              fill=EYE, width=0)
        else:
            for ex in (-7, 7):
                c.create_line(x + ex - 4, y - 5, x + ex + 4, y - 5, fill=EYE_SLEEP, width=2)

        if self.anim == 'yes':
            c.create_arc(x - 6, y - 3, x + 6, y + 5, start=200, extent=140, style='arc', outline=EYE, width=2)
        elif self.anim == 'no':
            c.create_arc(x - 6, y + 1, x + 6, y + 9, start=20, extent=140, style='arc', outline=EYE, width=2)
        else:
            c.create_line(x - 4, y + 3, x + 4, y + 3, fill=EYE if self.awake else EYE_SLEEP, width=2)

        if not self.awake:
            zcol_to = '#5b7a99' if self.lights else '#aab8d6'
            bg = DAY_BG if self.lights else NIGHT_BG
            for i in range(3):
                t = (now * 0.5 + i / 3) % 1
                c.create_text(x + 16 + t * 18, y - 26 - t * 26, text='z',
                              font=('Segoe UI', int(8 + t * 7), 'bold'), fill=mix(zcol_to, bg, t))


def main():
    root = tk.Tk()
    game = MarvinGame(root)
    threading.Thread(target=audio_worker, args=(game.events, game.stop_flag), daemon=True).start()
    root.mainloop()


if __name__ == '__main__':
    main()
