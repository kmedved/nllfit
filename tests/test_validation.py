import numpy as np
import pytest

from nllfit.validation import as_1d_float, validate_groups, validate_sample_weight


def test_as_1d_float_rejects_2d():
    with pytest.raises(ValueError, match="must be 1D"):
        as_1d_float("x", np.ones((5, 3)))


def test_as_1d_float_squeezes_column_vector():
    result = as_1d_float("x", np.ones((5, 1)))
    assert result.shape == (5,)


def test_as_1d_float_rejects_none():
    with pytest.raises(ValueError, match="must not be None"):
        as_1d_float("x", None)


def test_validate_sample_weight_negative():
    y = np.zeros(3)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_sample_weight(y, np.array([1.0, -1.0, 1.0]))


def test_validate_sample_weight_non_finite():
    y = np.zeros(3)
    with pytest.raises(ValueError, match="finite"):
        validate_sample_weight(y, np.array([1.0, np.inf, 1.0]))


def test_validate_sample_weight_length_mismatch():
    y = np.zeros(3)
    with pytest.raises(ValueError, match="length"):
        validate_sample_weight(y, np.array([1.0, 1.0]))


def test_validate_sample_weight_zero_sum():
    y = np.zeros(3)
    with pytest.raises(ValueError, match="positive sum"):
        validate_sample_weight(y, np.array([0.0, 0.0, 0.0]))


def test_validate_groups_rejects_2d():
    y = np.zeros(6)
    with pytest.raises(ValueError, match="must be 1D"):
        validate_groups(y, np.array([["a", "b"], ["c", "d"], ["e", "f"]]))
