"""Loaders for the paper's HMC runs (data/hmc, copied from the paper's analysis)."""
import pandas as pd

import style

D = style.DATA / "hmc"
TRUE_LAM = 400.0
TRUE_Q = 0.942
TRUE_WAB = 0.0786


def samples(rel):
    return pd.read_csv(D / rel / "samples.csv", index_col=0)


def best_fit(rel):
    return pd.read_csv(D / rel / "biases.csv", index_col=0).iloc[0]


def population(kind, n=16):
    """kind: 'qtpl' (textbook model) or 'qtplA' (includes the effect)."""
    return [samples(f"Anl_SNR175/population_x16/run_{kind}/run_{i}") for i in range(n)]
