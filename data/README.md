# data/

Inputs for the figures, copied from the paper's analysis (Bretz & Yu). Nothing here is edited.
Only what the build reads is included.

## hmc/ — Hamiltonian Monte Carlo samples

Each run folder has `samples.csv` (30,000 samples, in the order the sampler produced them, plus
`log_likelihood`) and, where present, `biases.csv` (the maximum-likelihood row).

| Folder | What it is | Used for |
|---|---|---|
| `Anl_SNR500/qtpl` | Textbook model fit to a signal with the nonlinear effect (A_nl = −0.03), SNR 500. Free: lam_cano, phi_coal, q, t_coal. f-mode tied to λ by the universal relation. | Ch. 6 sampler animation; ch. 7 posterior, SNR widget |
| `Anl_SNR500/qtplA_negA` | Same signal, model that includes A_nl (prior restricted to negative values). | Ch. 7 posterior |
| `Anl_SNR500/qtplo` | Textbook model with the f-mode frequency left free (no universal relation). | Ch. 7 "where does the error come from" |
| `Anl_SNR175/population_x16/run_qtpl/run_0 … run_15` | 16 events (different star masses), each SNR 175, textbook model. | Ch. 8 stacking animation |

Injected truth: tidal deformability 400, mass ratio 0.942, f-mode 0.0786 (canonical, SLy), A_nl = −0.03.

## paper_code/bns_tides/ — the paper's waveform code

Copied verbatim: `jax_waveform.py` (waveform and tidal phase, universal relations),
`run_parameters.py` (imported by it), `__init__.py`. `python/bns_import.py` loads it.
`jax_waveform.py` imports `ripple`, which is now published as `ripplegw`; `bns_import.py`
aliases the name. The only change from the original is that two local path strings, in code the
build never calls (a commented-out line and the `CE_Sn` helper), were replaced with generic
placeholders.

The original paper folder (PDF, LaTeX source, other runs, detector curves) is kept locally and is
listed in `.gitignore`.
