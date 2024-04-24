from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from scipy import stats
from sklearn.base import OutlierMixin, check_is_fitted
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from alchemy.curation.actions._base import BaseAction
from alchemy.report import CurationReport
from alchemy.types import VerbosityLevel


def modified_zscore(data: np.ndarray, consistency_correction: float = 1.4826):
    """
    The modified z score is calculated from the median absolute deviation (MAD).
    These values must be multiplied by a constant to approximate the standard deviation.

    The modified z score might be more robust than the standard z score because it relies
    on the median (MED) for calculating the z score.

    modified Z score = (X-MED) / (consistency_correction*MAD)

    """
    median = np.nanmedian(data)

    deviation_from_med = np.array(data) - median

    mad = np.nanmedian(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med / (consistency_correction * mad)
    return mod_zscore


class ZscoreOutlier(OutlierMixin):
    """
    Detect outliers by the absolute value of the Z-score.

    Uses a scikit-learn compatible interface.

    Args:
        threshold: If the absolute zscore is larger than this threshold, it is an inlier.
        use_modified_zscore: Flag to specify whether to use the normal or modified z-score.
    """

    _zscore: Optional[np.ndarray] = PrivateAttr(None)

    def __init__(self, threshold: float = 3, use_modified_zscore: bool = False):
        self.threshold = threshold
        self.use_modified_zscore = use_modified_zscore

    def fit(self, X: np.ndarray):
        """
        Computes the (potentially modified) z-scores for each of the observations.

        Args:
            X: The observations that we want to classify as inliers or outliers.
        """
        if self.use_modified_zscore:
            self._zscore = modified_zscore(X)
        else:
            self._zscore = stats.zscore(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: The observations that we want to classify as inliers or outliers.

        Returns:
            An array that for each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        check_is_fitted(self, attributes=["_zscore"])

        scores = np.absolute(self._zscore)
        decision_func = scores > self.threshold

        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func] = -1
        return is_inlier

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        First computes the z-scores, then predicts the inliers/outliers.

        Args:
            X: The observations that we want to classify as inliers or outliers.
        """
        self.fit(X)
        return self.predict(X)


class OutlierDetection(BaseAction):
    """
    Automatic detection of outliers.
    """

    method: Literal["iso", "lof", "svm", "ee", "zscore"]
    columns: List[str]
    prefix: str = "OUTLIER_"
    kwargs: Dict = Field(default_factory=dict)

    def run(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        detector_cls = _OUTLIER_METHODS[self.method]
        detector = detector_cls(**self.kwargs)

        for column in self.columns:
            values = dataset[column].values
            indices = np.flatnonzero(~np.isnan(values))

            in_ = values[indices].reshape(-1, 1)
            out_ = detector.fit_predict(in_)

            is_inlier = np.zeros_like(values, dtype=int)
            is_inlier[indices] = out_.flatten()

            dataset[self.get_column_name(column)] = is_inlier

        return dataset


_OUTLIER_METHODS = {
    "iso": IsolationForest,
    "lof": LocalOutlierFactor,
    "svm": OneClassSVM,
    "ee": EllipticEnvelope,
    "zscore": ZscoreOutlier,
}
