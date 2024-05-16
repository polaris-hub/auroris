from typing import Dict, List, Optional, Sequence

import pandas as pd

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import visualize_continuous_distribution


class ContinuousDistributionVisualization(BaseAction):
    """
    Visualize a continuous distribution

    Args:
        y_cols: List of columns for bioactivity for visualization
        log_scale: Whether visualize distribution in log scale.
        bins: The bin boundaries to color the area under the KDE curve.

    """

    y_cols: Optional[List[str]] = None
    log_scale: bool = False
    bins: Optional[Sequence[float]] = None

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        if report is not None:
            for y_col in self.y_cols:
                fig = visualize_continuous_distribution(
                    data=dataset[y_col], log_scale=self.log_scale, bins=self.bins
                )
                report.log_image(fig, title=f"Data distribution - {y_col}")

        return dataset
