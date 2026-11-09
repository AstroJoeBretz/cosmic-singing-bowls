"""Timeline of the chapter 1 animation, shared with its audio (fig_sound.py)."""
N = 330       # frames of inspiral
HOLD = 40     # frames held on the final pose
FPS = 30
DURATION_S = (N + HOLD) / FPS
CONTACT_S = (N - 1) / FPS     # the stars reach their final pose (contact) on the last inspiral frame
