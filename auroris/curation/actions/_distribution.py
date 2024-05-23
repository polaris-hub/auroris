from typing import Dict, List, Literal, Optional, Sequence

import pandas as pd
from loguru import logger

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import visualize_continuous_distribution


class ContinuousDistributionVisualization(BaseAction):
    """
    Visualize one or more continuous distribution(s).

    See [`auroris.visualization.visualize_continuous_distribution`][] for the docs of the
    `log_scale` and `bins` attributes

    Attributes:
        y_cols: The columns whose distributions should be visualized.
    """

    name: Literal["distribution"] = "distribution"

    y_cols: List[str]
    log_scale: bool = False
    bins: Optional[Sequence[float]] = None

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        if report is None:
            logger.warning("No report provided. Skipping visualization.")

        if report is not None:
            for y_col in self.y_cols:
                fig = visualize_continuous_distribution(
                    data=dataset[y_col], log_scale=self.log_scale, bins=self.bins
                )
                report.log_image(fig, title=f"Data distribution - {y_col}")

        return dataset
