from typing import Dict, List, Optional

import pandas as pd
from pydantic import Field

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import visualize_continuous_distribution


class ContinuousDistributionVisualization(BaseAction):
    """
    Visualize a continuous distribution
    """

    y_cols: Optional[List[str]] = None
    log_scale: bool = False
    kwargs: Dict = Field(default_factory=dict)

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
                    data=dataset[y_col], label_name=y_col, log_scale=self.log_scale
                )
                report.log_image(fig, title=f"Data distribution - {y_col}")

        return dataset
