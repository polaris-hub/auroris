from typing import Dict, List, Optional
import pandas as pd
from pydantic import Field
import numpy as np

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import detailed_distributions_plots


class DataDistribution(BaseAction):
    """
    Access the data distribution
    """

    y_cols: Optional[List[str]] = None
    log_scale: bool = False
    kwargs: Dict = Field(default_factory=dict)

    def transform(
        self,
        dataset: pd.DataFrame,
        discretizers: Optional[callable] = None,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        if report is not None:
            for y_col in self.y_cols:
                discretizer = discretizers.get(y_col)
                sections = []
                if discretizer is not None:
                    low = -np.inf
                    high = np.inf
                    for i, threshold in enumerate(discretizer.thresholds + [high]):
                        X = dataset[f"{discretizer.prefix}{y_col}"].values
                        if discretizer.label_order == "descending":
                            i = len(discretizer.thresholds) - i
                        pct = 100 * sum(X == i) / len(X)
                        sections.append(
                            {
                                "label": f"{discretizer.prefix}{y_col} = {i}: {pct:.1f} %",
                                "start": low,
                                "end": threshold,
                                "pct": pct,
                            }
                        )
                        low = threshold
                fig = detailed_distributions_plots(
                    data=dataset[y_col], label_name=y_col, sections=sections, log_scale=self.log_scale
                )
                report.log_image(fig, title=f"Data distribution - {y_col}")

        return dataset
