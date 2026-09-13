from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from turn_taking.config import Config, TrainConfig
from turn_taking.dataset import Dataset
from turn_taking.features import ContextFeatures, FeatureExtractor
from turn_taking.infer import TurnTakingPredictor
from turn_taking.train import (
    build_pipeline,
    cross_validate,
    load_model,
    save_model,
    train_test_evaluate,
)


def _toy_dataset(config: Config, n_per_class: int = 20, n_groups: int = 4) -> Dataset:
    """A tiny, trivially-separable dataset: features are pure noise around two means."""
    rng = np.random.default_rng(0)
    extractor = FeatureExtractor(config)
    n_features = extractor.n_features

    X0 = rng.normal(loc=0.0, scale=0.1, size=(n_per_class, n_features))
    X1 = rng.normal(loc=5.0, scale=0.1, size=(n_per_class, n_features))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    groups = np.array([f"speaker{i % n_groups}" for i in range(len(y))])
    clip_ids = [f"clip{i}" for i in range(len(y))]
    raw_labels = ["continue" if label == 0 else "interrupt" for label in y]
    return Dataset(X=X, y=y, groups=groups, clip_ids=clip_ids, raw_labels=raw_labels, feature_names=extractor.feature_names)


def test_build_pipeline_rejects_unknown_model(config: Config):
    with pytest.raises(ValueError):
        build_pipeline(TrainConfig(model="not_a_real_model"))


def test_train_test_evaluate_separable_data_is_near_perfect(config: Config):
    dataset = _toy_dataset(config)
    result = train_test_evaluate(dataset, config)
    assert result.metrics.accuracy >= 0.9
    # cross_validate clamps n_splits to the number of distinct groups (4 here).
    assert len(result.cv_metrics) == min(config.train.n_splits, 4)


def test_cross_validate_requires_multiple_groups(config: Config):
    dataset = _toy_dataset(config, n_groups=1)
    with pytest.raises(ValueError, match="grouped cross-validation"):
        cross_validate(dataset, config)


def test_train_test_evaluate_single_group_falls_back(config: Config):
    dataset = _toy_dataset(config, n_groups=1)
    result = train_test_evaluate(dataset, config)
    assert result.cv_metrics == []
    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_train_test_evaluate_rejects_tiny_dataset(config: Config):
    extractor = FeatureExtractor(config)
    dataset = Dataset(
        X=np.zeros((1, extractor.n_features)),
        y=np.array([0]),
        groups=np.array(["a"]),
        clip_ids=["c1"],
        raw_labels=["continue"],
        feature_names=extractor.feature_names,
    )
    with pytest.raises(ValueError):
        train_test_evaluate(dataset, config)


def test_save_and_load_model_roundtrip(tmp_path: Path, config: Config):
    dataset = _toy_dataset(config)
    result = train_test_evaluate(dataset, config)
    model_path = tmp_path / "model.joblib"
    save_model(result, model_path)

    assert model_path.exists()
    assert model_path.with_suffix(".metrics.json").exists()

    loaded = load_model(model_path)
    assert loaded.feature_names == result.feature_names
    predictions = loaded.pipeline.predict(dataset.X)
    assert predictions.shape == dataset.y.shape


def test_load_model_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_predictor_predict_proba_in_unit_interval(tmp_path: Path, config: Config):
    dataset = _toy_dataset(config)
    result = train_test_evaluate(dataset, config)
    model_path = tmp_path / "model.joblib"
    save_model(result, model_path)

    predictor = TurnTakingPredictor(model_path)
    signal = 0.1 * np.random.default_rng(3).standard_normal(config.audio.window_samples)
    probability = predictor.predict_proba(signal, ContextFeatures())
    assert 0.0 <= probability <= 1.0
