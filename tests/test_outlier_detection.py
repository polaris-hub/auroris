import numpy as np
import pandas as pd
import pytest

from auroris.curation.actions import OutlierDetection
from auroris.curation.functional import detect_outliers


@pytest.mark.parametrize("method", ["iso", "lof", "svm", "ee", "zscore"])
def test_outlier_detection(method, dataset):
    action = OutlierDetection(method=method, columns=["outlier_column"])

    df = action.transform(dataset)
    assert f"{action.prefix}outlier_column" in df.columns

    outliers = df[df[f"{action.prefix}outlier_column"] == 1]

    # It's hard to conceive a toy example that works for all methods
    # This is not the goal of this test case. Except for the z-score (which we tested separately)
    # the implementation of the outlier detection comes from Scikit-learn and we can just do a sanity check.
    assert len(outliers) >= 2
    assert 0 in outliers.index
    assert (len(dataset) - 1) in outliers.index


@pytest.mark.parametrize("use_modified_zscore", [True, False])
def test_zscore_outlier_detection(use_modified_zscore, dataset):
    action = OutlierDetection(
        method="zscore",
        columns=["outlier_column"],
        kwargs={
            "use_modified_zscore": use_modified_zscore,
            "threshold": 4.5,
        },
    )

    df = action.transform(dataset)
    assert f"{action.prefix}outlier_column" in df.columns

    outliers = df[df[f"{action.prefix}outlier_column"] == 1]

    assert len(outliers) == 2
    assert 0 in outliers.index
    assert (len(dataset) - 1) in outliers.index


def test_check_outlier_zscore():
    data = pd.DataFrame(np.random.normal(0, 0.1, size=100), columns=["data_col"])
    num_outlier = 5
    data.loc[: num_outlier - 1, "data_col"] = 10
    is_outlier = detect_outliers(X=data["data_col"].values, method="zscore")
    assert is_outlier.sum() == num_outlier
