"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from turn_taking.config import Config


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
