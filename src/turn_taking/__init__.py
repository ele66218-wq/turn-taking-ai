"""Turn-taking AI: estimate when a spoken-dialogue system should stop talking.

The package is organised around one question:

    Given the last ~1 second of user audio (plus timing context),
    what is P(the user wants the floor)?

Modules
-------
config      : typed configuration loaded from YAML.
labels      : annotation schema and labels.csv I/O.
audio_io    : WAV I/O and microphone capture.
features    : acoustic feature extraction (numpy/scipy only).
dataset     : labels.csv -> feature matrix.
metrics     : turn-taking specific metrics (false stop / missed interruption / latency).
train       : scikit-learn baseline training.
infer       : model loading and single-window inference.
controller  : CONTINUE / PAUSE / YIELD state machine with hysteresis.
realtime    : microphone loop wiring everything together.
synthetic   : synthetic dataset generator so the pipeline runs without a mic.
"""

__version__ = "0.1.0"
