import numpy as np
import pandas as pd

from auroris.curation.functional import detect_streoisomer_activity_cliff


def test_identify_stereoisomers_with_activity_cliff():
    vals = list(np.random.randint(0, 10, 50)) * 2 + np.random.normal(0, 0.01, 100)
    groups = list(range(50)) * 2
    num_cliff = 10
    index_cliff = np.random.randint(0, 50, num_cliff)
    vals[index_cliff] = 1000

    data = pd.DataFrame({"data_col": vals, "groupby_col": groups})
    df = detect_streoisomer_activity_cliff(
        dataset=data,
        stereoisomer_id_col="groupby_col",
        threshold=3,
        y_cols=["data_col"],
    )
    # check if identifed ids are correct
    ids = df[df["AC_data_col"]]["groupby_col"].unique()
    assert set(ids) == set(index_cliff)
