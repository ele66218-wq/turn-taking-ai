"""Baseline model training: scikit-learn on top of the handcrafted features.

Deliberately simple (RandomForest / LogisticRegression on ~90 features) --
this is the "stage 1" model from the project's design discussion. Stage 2
(wav2vec2/HuBERT embeddings + a small MLP) can plug into the same
``Dataset``/``metrics`` contract without touching this module's callers.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config, TrainConfig
from .dataset import Dataset
from .metrics import TurnTakingMetrics, compute_metrics

__all__ = ["TrainResult", "build_pipeline", "cross_validate", "train_test_evaluate", "save_model", "load_model"]


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: TurnTakingMetrics
    cv_metrics: list[TurnTakingMetrics]
    feature_names: list[str]
    config: Config


def build_pipeline(train_cfg: TrainConfig) -> Pipeline:
    """Build the (scaler -> classifier) pipeline for the given training config.

    Raises
    ------
    ValueError
        If ``train_cfg.model`` names an unsupported classifier.
    """
    if train_cfg.model == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=train_cfg.n_estimators,
            max_depth=train_cfg.max_depth,
            class_weight=train_cfg.class_weight,
            random_state=train_cfg.random_state,
            n_jobs=-1,
        )
        # Trees do not need feature scaling, but keeping the scaler in the
        # pipeline means both model types share one code path and one
        # serialized artifact shape.
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    if train_cfg.model == "logistic_regression":
        classifier = LogisticRegression(
            class_weight=train_cfg.class_weight,
            random_state=train_cfg.random_state,
            max_iter=2000,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    raise ValueError(f"unsupported train.model: {train_cfg.model!r}")


def cross_validate(dataset: Dataset, config: Config) -> list[TurnTakingMetrics]:
    """Grouped k-fold cross-validation (grouped by speaker) for an honest estimate.

    Falls back to fewer folds if there are not enough distinct speaker groups,
    and raises rather than silently producing a single degenerate fold.

    Raises
    ------
    ValueError
        If fewer than 2 distinct groups are present (grouped CV is impossible).
    """
    n_groups = len(set(dataset.groups.tolist()))
    if n_groups < 2:
        raise ValueError(
            "grouped cross-validation needs >= 2 distinct speakers/groups, "
            f"found {n_groups}. Add more 'speaker' values in labels.csv, or "
            "call train_test_evaluate directly for a quick single-split check."
        )
    n_splits = min(config.train.n_splits, n_groups)
    splitter = GroupKFold(n_splits=n_splits)

    results: list[TurnTakingMetrics] = []
    for train_idx, test_idx in splitter.split(dataset.X, dataset.y, groups=dataset.groups):
        pipeline = build_pipeline(config.train)
        pipeline.fit(dataset.X[train_idx], dataset.y[train_idx])
        predictions = pipeline.predict(dataset.X[test_idx])
        results.append(compute_metrics(dataset.y[test_idx], predictions))
    return results


def train_test_evaluate(dataset: Dataset, config: Config) -> TrainResult:
    """Fit on a grouped train/test split, evaluate, then refit on all data.

    The returned pipeline is refit on the *entire* dataset (common practice
    once the held-out metrics are recorded) so the saved model uses every
    labeled clip; ``metrics`` and ``cv_metrics`` still reflect held-out
    performance only.

    Raises
    ------
    ValueError
        If the dataset has fewer than 2 samples, or a grouped split cannot
        separate at least one clip into the test set.
    """
    if len(dataset) < 2:
        raise ValueError(f"need at least 2 samples to train, got {len(dataset)}")

    n_groups = len(set(dataset.groups.tolist()))
    if n_groups >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.train.test_size, random_state=config.train.random_state
        )
        train_idx, test_idx = next(splitter.split(dataset.X, dataset.y, groups=dataset.groups))
        cv_metrics = cross_validate(dataset, config)
    else:
        # Single-speaker dataset (e.g. a first personal recording session):
        # fall back to a plain stratified-by-index split and skip grouped CV.
        rng = np.random.default_rng(config.train.random_state)
        order = rng.permutation(len(dataset))
        n_test = max(1, int(round(len(dataset) * config.train.test_size)))
        test_idx, train_idx = order[:n_test], order[n_test:]
        cv_metrics = []

    pipeline = build_pipeline(config.train)
    pipeline.fit(dataset.X[train_idx], dataset.y[train_idx])
    predictions = pipeline.predict(dataset.X[test_idx])
    holdout_metrics = compute_metrics(dataset.y[test_idx], predictions)

    final_pipeline = build_pipeline(config.train)
    final_pipeline.fit(dataset.X, dataset.y)

    return TrainResult(
        pipeline=final_pipeline,
        metrics=holdout_metrics,
        cv_metrics=cv_metrics,
        feature_names=dataset.feature_names,
        config=config,
    )


def save_model(result: TrainResult, path: str | Path) -> None:
    """Save the fitted pipeline plus metadata needed to reload it safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "pipeline": result.pipeline,
        "feature_names": result.feature_names,
        "config": result.config.to_dict(),
        "metrics": result.metrics.to_dict(),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    joblib.dump(payload, path)

    report_path = path.with_suffix(".metrics.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "holdout": result.metrics.to_dict(),
                "cross_validation": [m.to_dict() for m in result.cv_metrics],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


@dataclass
class LoadedModel:
    pipeline: Pipeline
    feature_names: list[str]
    config: Config
    sklearn_version: str


def load_model(path: str | Path) -> LoadedModel:
    """Load a model saved by :func:`save_model`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If the file was saved with a materially different scikit-learn
        version's major.minor (pipelines are not guaranteed compatible across
        versions) -- reported as a warning, not a hard failure, since joblib
        itself does not enforce this.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    payload = joblib.load(path)

    saved_version = payload.get("sklearn_version", "unknown")
    current_major_minor = ".".join(sklearn.__version__.split(".")[:2])
    saved_major_minor = ".".join(str(saved_version).split(".")[:2])
    if saved_major_minor != "unknown" and saved_major_minor != current_major_minor:
        print(
            f"[turn_taking] warning: model was saved with scikit-learn {saved_version}, "
            f"current is {sklearn.__version__}. Predictions may be inconsistent.",
            file=sys.stderr,
        )

    from .config import Config as _Config  # local import to avoid a cycle at module load

    return LoadedModel(
        pipeline=payload["pipeline"],
        feature_names=payload["feature_names"],
        config=_Config(**_rebuild_config_kwargs(payload["config"])),
        sklearn_version=str(saved_version),
    )


def _rebuild_config_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    from .config import AudioConfig, ControllerConfig, FeatureConfig, TrainConfig as _TrainConfig

    return {
        "audio": AudioConfig(**raw["audio"]),
        "features": FeatureConfig(**raw["features"]),
        "controller": ControllerConfig(**raw["controller"]),
        "train": _TrainConfig(**raw["train"]),
    }
