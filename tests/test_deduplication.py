import numpy as np
import pandas as pd

from auroris.curation.functional import deduplicate


def test_deduplicate():
    ids = np.array(range(100))
    max_ind = max(ids)

    num_dup = 5
    for index in range(num_dup):
        ids[max_ind - index] = index

    data_col_1 = np.random.normal(1, 0.01, 100)
    data_col_2 = np.random.normal(2, 0.01, 100)
    data_col_3 = np.random.normal(3, 0.01, 100)

    data = pd.DataFrame(
        {"data_col_1": data_col_1, "data_col_2": data_col_2, "data_col_3": data_col_3, "ids": ids}
    )

    merged = deduplicate(
        dataset=data,
        y_cols=["data_col_1", "data_col_2", "data_col_3"],
        deduplicate_on=["ids"],
        method="median",
    )

    # check the data points been merged
    assert data.shape[0] == merged.shape[0] + num_dup

    # check the merged values are correct
    for index in range(num_dup):
        assert data.loc[[index, max_ind - index], "data_col_1"].median() == merged.loc[index, "data_col_1"]
        assert data.loc[[index, max_ind - index], "data_col_2"].median() == merged.loc[index, "data_col_2"]
        assert data.loc[[index, max_ind - index], "data_col_3"].median() == merged.loc[index, "data_col_3"]
