from typing import Dict, List, Optional
import pandas as pd
from pydantic import Field, PrivateAttr


from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.curation.actions._discretize import Discretization
from auroris.visualization import detailed_distributions_plots


class DataDistribution(BaseAction):
    """
    Access the data distribution
    """

    y_cols: Optional[List[str]] = None
    log_scale_mapping: Dict[str, bool] = None
    kwargs: Dict = Field(default_factory=dict)

    def transform(
        self,
        dataset: pd.DataFrame,
        discretizer: Optional[callable] = None,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        if report is not None:
            fig = detailed_distributions_plots(df=dataset, label_names=self.y_cols, thresholds={""})
            report.log_image(fig, title="Data distribution")

        return dataset
