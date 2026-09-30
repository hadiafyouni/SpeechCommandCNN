"""
Console listener: always listening for the wake word "marvin", then records
and classifies one command, then goes back to sleep. Nothing is classified
unless the wake word is heard first.

Run:  python listener.py        (Ctrl+C to quit)

The status line shows the live mic level and wake-word probability — useful
for tuning WAKE_THRESHOLD / SPEECH_RMS_GATE in config.py.
"""

import sys

import numpy as np

import config
import voice_engine as ve


def listen_for_command():
    print('\n>>> AWAKE — say your command now <<<')
    ve.flush_queue()   # drop the tail of "marvin" so we only hear the command
    word, conf = ve.classify(ve.next_samples(ve.window_len))
    print('-->', ve.describe(word, conf))


def main():
    ve.warm_up()
    print('\n--- Wake-word Speech Command Listener ---')
    print('Microphone:', ve.MIC_NAME, '@', ve.MIC_RATE, 'Hz')
    print(f'Say "{config.WAKE_WORD}" to wake it up, then one of:', config.COMMAND_WORDS)
    print('Ctrl+C to quit.\n')

    window = np.zeros(ve.window_len, dtype=np.float32)   # the last second of audio
    fresh = 0          # samples of real audio in the window since start / last command
    since_check = 0
    streak = 0

    with ve.open_mic_stream():
        try:
            while True:
                chunk = ve.audio_q.get()
                window = np.concatenate([window[len(chunk):], chunk])
                fresh += len(chunk)
                since_check += len(chunk)

                # only judge a window of 100% fresh audio (not startup zeros or
                # the end of the previous command), once per hop
                if fresh < ve.window_len or since_check < ve.hop_len:
                    continue
                since_check = 0

                level, prob = ve.wake_probability(window)
                streak = streak + 1 if prob > config.WAKE_THRESHOLD else 0
                sys.stdout.write(f'\r listening... level: {level:.4f}  wake-word probability: {prob:.2f} '
                                 f'(streak: {streak}, mic overflows: {ve.overflow_count})   ')
                sys.stdout.flush()

                if streak >= config.CONSECUTIVE_HOPS_REQUIRED:
                    listen_for_command()
                    print(f'--- back to sleep, say "{config.WAKE_WORD}" again ---\n')
                    ve.flush_queue()
                    streak = fresh = 0
        except KeyboardInterrupt:
            print('\nStopping.')


if __name__ == '__main__':
    main()
