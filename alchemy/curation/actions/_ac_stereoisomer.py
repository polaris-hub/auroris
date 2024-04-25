from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alchemy.curation.actions._base import BaseAction
from alchemy.curation.actions._outlier import modified_zscore
from alchemy.report import CurationReport
from alchemy.types import VerbosityLevel
from alchemy.utils import is_regression


def detect_streoisomer_activity_cliff(
    dataset: pd.DataFrame,
    stereoisomer_id_col: str,
    y_cols: List[str],
    threshold: float = 1.0,
    prefix: str = "AC_",
):
    groups = []
    ac_cols = {y_col: [] for y_col in y_cols}

    dataset = dataset.reset_index(drop=True)

    for _, group in dataset.groupby(stereoisomer_id_col):
        for y_col in y_cols:
            if is_regression(group[y_col].values):
                # In regression, we use the difference between the z-scores
                zscores = modified_zscore(dataset[y_col].values)[group.index]
                ac = (zscores.max() - zscores.min()) > threshold
            else:
                # For classification, we use the number of unique classes
                ac = len(np.unique(group[y_col].values)) > 1
            ac_cols[y_col].extend([ac] * len(group))

        groups.append(group)

    dataset = pd.concat(groups)
    for y_col in y_cols:
        dataset[f"{prefix}{y_col}"] = ac_cols[y_col]

    return dataset.sort_index()


class StereoIsomer(BaseAction):
    """
    Automatic detection of outliers.
    """

    stereoisomer_id_col: str
    y_cols: List[str]
    threshold: float = 2.0
    prefix: str = "AC_"

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        return detect_streoisomer_activity_cliff(
            dataset=dataset,
            stereo_column=self.stereo_column,
            y_cols=self.y_cols,
            threshold=self.threshold,
            prefix=self.prefix,
        )
