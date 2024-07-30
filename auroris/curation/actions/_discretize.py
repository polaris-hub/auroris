from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import check_array

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import visualize_continuous_distribution


def discretize(
    X: np.ndarray,
    thresholds: Union[np.ndarray, list],
    inplace: bool = False,
    allow_nan: bool = True,
    label_order: Literal["ascending", "descending"] = "ascending",
) -> np.ndarray:
    """
    Thresholding of array-like or scipy.sparse matrix into binary or multiclass labels.

    Args:
        X : The data to discretize, element by element.
            scipy.sparse matrices should be in CSR or CSC format to avoid an
            un-necessary copy.

        thresholds: Interval boundaries that include the right bin edge.

        inplace: Set to True to perform inplace discretization and avoid a copy
            (if the input is already a numpy array or a scipy.sparse CSR / CSC
            matrix and if axis is 1).

        allow_nan: Set to True to allow nans in the array for discretization. Otherwise,
            an error will be raised instead.

        label_order: The continuous values are discretized to labels 0, 1, 2, .., N with respect to given
            threshold bins [threshold_1, threshold_2,.., threshould_n].
            When set to 'ascending', the class label is in ascending order with the threshold
            bins that `0` represents negative class or lower class, while 1, 2, 3 are for higher classes.
            When set to 'descending' the class label is in ascending order with the threshold bins.
            Sometimes the positive labels are on the left side of provided threshold.
            E.g. For binarization with threshold [0.5],  the positive label is defined
            by`X < 0.5`. In this case, `label_order` should be `descending`.

    Returns:
        X_tr: The transformed data.
    """
    if label_order not in ["ascending", "descending"]:
        raise ValueError(
            f"{label_order} is not a valid label_order. Choose from 'ascending' or 'descending'."
        )

    X = check_array(
        X,
        accept_sparse=["csr", "csc"],
        copy=not inplace,
        force_all_finite="allow-nan" if allow_nan else True,
        ensure_2d=False,
    )

    nan_idx = np.isnan(X)

    thresholds = thresholds
    binarize = True if len(thresholds) == 1 else False

    if label_order == "descending":
        thresholds = np.flip(thresholds)
    X = np.digitize(X, thresholds)

    if allow_nan:
        X = X.astype(np.float64)
        X[nan_idx] = np.nan
    if binarize and label_order == "descending":
        X = 1 - X
    return X


class Discretization(BaseAction):
    """
    Thresholding bioactivity columns to binary or multiclass labels.

    See [`auroris.curation.functional.discretize`][] for the docs of the
    `thresholds`, `inplace`, `allow_nan` and `label_order` attributes

    Attributes:
        input_column: The column to discretize.
        log_scale: Whether a visual depiction of the discretization should be on a log scale.
    """

    name: Literal["discretize"] = "discretize"
    prefix: str = "CLS_"

    input_column: str
    thresholds: List[float]

    inplace: bool = False
    allow_nan: bool = True
    label_order: Literal["ascending", "descending"] = "ascending"
    log_scale: bool = False

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        X = dataset[self.input_column].values
        X = discretize(
            X,
            thresholds=self.thresholds,
            inplace=self.inplace,
            allow_nan=self.allow_nan,
            label_order=self.label_order,
        )

        fig = visualize_continuous_distribution(
            data=dataset[self.input_column].values,
            log_scale=self.log_scale,
            bins=self.thresholds,
        )
        report.log_image(fig, title=f"Data distribution - {self.input_column}")

        column_name = self.get_column_name(self.input_column)
        dataset[column_name] = X

        if report is not None:
            report.log_new_column(column_name)

        return dataset
