from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from run_null_models import empirical_upper_tail_p


def test_empirical_p_never_zero() -> None:
    null_values = np.array([0.1, 0.2, 0.3])
    assert empirical_upper_tail_p(null_values, 1.0) == 0.25


def test_empirical_p_is_one_when_every_null_is_extreme() -> None:
    null_values = np.array([0.8, 0.9, 1.0])
    assert empirical_upper_tail_p(null_values, 0.5) == 1.0


def test_empirical_p_requires_simulations() -> None:
    try:
        empirical_upper_tail_p(np.array([]), 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an empty null sample.")
