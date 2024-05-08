from typing import Dict, List, Optional

import datamol as dm
import numpy as np
import pandas as pd

from auroris.curation.actions._base import BaseAction
from auroris.curation.actions._outlier import modified_zscore
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.utils import is_regression


def detect_streoisomer_activity_cliff(
    dataset: pd.DataFrame,
    stereoisomer_id_col: str,
    y_cols: List[str],
    threshold: float = 1.0,
    prefix: str = "AC_",
):
    dataset_ori = dataset.copy(deep=True)
    ac_cols = {y_col: [] for y_col in y_cols}
    group_index_list = np.array(
        [group.index.values for _, group in dataset.groupby(stereoisomer_id_col, sort=False)]
    )
    for y_col in y_cols:
        is_reg = is_regression(dataset[y_col].dropna().values)
        if is_reg:
            y_zscores = modified_zscore(dataset[y_col].values)

        for group_index in group_index_list:
            group = dataset.iloc[group_index, :]
            if len(group) == 1:
                ac = None
            else:
                if is_reg:
                    # In regression, we use the difference between the z-scores
                    zscores = y_zscores[group.index]
                    ac = (np.nanmax(zscores) - np.nanmin(zscores)) > threshold
                else:
                    # For classification, we use the number of unique classes
                    ac = len(np.unique(group[y_col].values)) > 1
            ac_cols[y_col].extend([ac] * len(group))

    for y_col in y_cols:
        rows = group_index_list.flatten()
        dataset_ori.loc[rows, f"{prefix}{y_col}"] = np.array(ac_cols[y_col]).astype(bool)

    return dataset_ori


class StereoIsomerACDetection(BaseAction):
    """
    Automatic detection of outliers.
    """

    stereoisomer_id_col: str
    y_cols: List[str]
    threshold: float = 2.0
    prefix: str = "AC_"
    mol_col: str = "MOL_smiles"

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        dataset = detect_streoisomer_activity_cliff(
            dataset=dataset,
            stereoisomer_id_col=self.stereoisomer_id_col,
            y_cols=self.y_cols,
            threshold=self.threshold,
            prefix=self.prefix,
        )

        if report is not None:
            for col in self.y_cols:
                col_with_prefix = self.get_column_name(col)
                report.log_new_column(col_with_prefix)

                has_cliff = dataset[col_with_prefix].notna()
                num_cliff = has_cliff.sum()

                if num_cliff > 0:
                    report.log(
                        f"Found {num_cliff} activity cliffs among stereoisomers "
                        f"with respect to the {col} column."
                    )
                    to_plot = dataset.loc[has_cliff, self.mol_col]
                    legends = (col + dataset.loc[has_cliff, col].astype(str)).tolist()

                    image = dm.to_image([dm.to_mol(s) for s in to_plot], legends=legends, use_svg=False)
                    report.log_image(image)

                else:
                    report.log(
                        "Found no activity cliffs among stereoisomers with respect to the {col} column."
                    )
        return dataset
