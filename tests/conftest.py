import datamol as dm
import numpy as np
import pytest


@pytest.fixture(scope="function")
def dataset():
    data = dm.data.freesolv()

    data["outlier_column"] = np.random.normal(0, 1, len(data))
    data.loc[data.index[-1], "outlier_column"] = 5
    data.loc[data.index[0], "outlier_column"] = -5
    data.loc[data.index[1], "outlier_column"] = np.nan
    return data
