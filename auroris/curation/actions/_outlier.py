from typing import Any, Dict, List, Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from scipy import stats
from sklearn.base import OutlierMixin, check_is_fitted
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import visualize_distribution_with_outliers

OutlierDetectionMethod: TypeAlias = Literal["iso", "lof", "svm", "ee", "zscore"]


def detect_outliers(X: np.ndarray, method: OutlierDetectionMethod = "zscore", **kwargs: Any):
    """Functional interface for detecting outliers

    Args:
        X: The observations that we want to classify as inliers or outliers.
        method: The method to use for outlier detection.
        **kwargs: Keyword arguments for the outlier detection method.
    """

    if X.ndim != 1:
        raise ValueError("X must be a 1D array for outlier detection.")

    detector_cls = _OUTLIER_METHODS[method]
    detector = detector_cls(**kwargs)
    indices = np.flatnonzero(~np.isnan(X))

    in_ = X[indices].reshape(-1, 1)
    out_ = detector.fit_predict(in_)

    is_inlier = np.full_like(X, np.nan)
    is_inlier[indices] = out_.flatten()

    is_outlier = is_inlier == -1
    return is_outlier


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

    method: OutlierDetectionMethod
    columns: List[str]
    prefix: str = "OUTLIER_"
    kwargs: Dict = Field(default_factory=dict)

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        for column in self.columns:
            values = dataset[column].values
            is_outlier = detect_outliers(values, self.method, **self.kwargs)

            is_outlier_col_label = self.get_column_name(column)
            dataset[is_outlier_col_label] = is_outlier

            if report is not None:
                report.log_new_column(is_outlier_col_label)

                fig = visualize_distribution_with_outliers(values=values, is_outlier=is_outlier)
                report.log_image(fig)

        return dataset


_OUTLIER_METHODS: Dict[OutlierDetectionMethod, OutlierMixin] = {
    "iso": IsolationForest,
    "lof": LocalOutlierFactor,
    "svm": OneClassSVM,
    "ee": EllipticEnvelope,
    "zscore": ZscoreOutlier,
}
