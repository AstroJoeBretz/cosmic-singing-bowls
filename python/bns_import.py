"""Import the paper's waveform code without modifying it.

The code lives in data/paper_code/bns_tides (copied verbatim from the paper's
analysis). It imports `ripple`, which is now published as `ripplegw`, so we alias
the module. Only the waveform functions are used; `CE_Sn` (which has a hard-coded
path) is never called.
"""
import sys

import numpy as np
from jax import config

config.update("jax_enable_x64", True)  # the paper code needs float64 (cgs units)

import ripplegw
import ripplegw.waveforms

sys.modules.setdefault("ripple", ripplegw)
sys.modules.setdefault("ripple.waveforms", ripplegw.waveforms)

from style import DATA  # noqa: E402

sys.path.insert(0, str(DATA / "paper_code"))
from bns_tides import jax_waveform as jw  # noqa: E402,F401
