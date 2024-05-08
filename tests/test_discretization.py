import numpy as np
import pytest

from auroris.curation.functional import discretize


def test_discretizer():
    X = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    thresholds_binary = [0.5]
    thresholds_multiclass = [0, 1]

    values_binary = discretize(X=X, thresholds=thresholds_binary)
    assert np.array_equal(values_binary, np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]]))

    values_binary_r = discretize(X=X, thresholds=thresholds_binary, label_order="descending")
    assert np.array_equal(values_binary_r, np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]))

    values_multiclass = discretize(X=X, thresholds=thresholds_multiclass)
    assert np.array_equal(values_multiclass, np.array([[2, 0, 2], [2, 1, 1], [1, 2, 0]]))

    values_multiclass_r = discretize(X=X, thresholds=thresholds_multiclass, label_order="descending")
    assert np.array_equal(values_multiclass_r, np.array([[0, 2, 0], [0, 1, 1], [1, 0, 2]]))

    with pytest.raises(ValueError):
        discretize(X=X, thresholds=thresholds_multiclass, label_order="WrongValue")
