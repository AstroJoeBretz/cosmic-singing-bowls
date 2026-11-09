"""Timeline and response curve shared by the chapter 4 animation and its audio."""
import numpy as np

N, HOLD, FPS = 300, 45, 30
ZETA = 0.07
SING_THRESHOLD = 0.35     # "it sings!" shows while the response is above this fraction of its peak (~2.1 s)
AUDIO_POWER = 2.0         # soundtrack loudness = (response / peak) ** AUDIO_POWER: much quieter off resonance


def response(W):
    return 1.0 / np.sqrt((1 - W ** 2) ** 2 + (2 * ZETA * W) ** 2)
