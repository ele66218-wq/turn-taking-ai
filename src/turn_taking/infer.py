"""Single-window inference: audio in, P(user wants the floor) out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .features import ContextFeatures, FeatureExtractor
from .train import LoadedModel, load_model


class TurnTakingPredictor:
    """Wraps a loaded model + feature extractor for repeated single-window calls."""

    def __init__(self, model_path: str | Path):
        self.loaded: LoadedModel = load_model(model_path)
        self.extractor = FeatureExtractor(self.loaded.config)
        self._positive_index = self._resolve_positive_index()

    @property
    def config(self) -> Config:
        return self.loaded.config

    def _resolve_positive_index(self) -> int:
        classes = list(self.loaded.pipeline.classes_)
        if 1 not in classes:
            raise RuntimeError(f"model classes {classes} do not contain the positive class 1")
        return classes.index(1)

    def predict_proba(
        self, signal: NDArray[np.float64], context: ContextFeatures | None = None
    ) -> float:
        """P(user wants the floor) for one window of audio (already right-aligned)."""
        vector = self.extractor(signal, context).reshape(1, -1)
        probabilities = self.loaded.pipeline.predict_proba(vector)[0]
        return float(probabilities[self._positive_index])
