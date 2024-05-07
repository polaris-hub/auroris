from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats
import matplotlib.pyplot as plt

from auroris.visualization.utils import create_figure


def detailed_distributions_plots(
    data: pd.DataFrame, label_name: str, sections: Optional[List[dict]] = None, log_scale: bool = False
):
    """
    KDE plot the distribution of the column in `data` with colored sections under the KDE curve.
    Args:
        df: A dataframe with binarized readouts only. NaN are allowed.
        label_name: Name of the labels (same order as the columns in `df`). If not set the name of the columns are used.
        log_scale: Whether set axis scale(s) to log.
    """
    # Create a KDE plot without filling
    fig = sns.kdeplot(data, color="black", linewidth=1.5, label="KDE Curve", log_scale=log_scale)

    # Calculate KDE values for filling sections
    try:
        kde_values = sns.kdeplot(data).get_lines()[0].get_data()
    except Exception as e:
        logger.exception(e)
        if log_scale:
            logger.exception(
                "The current error is likely due to the `log_scale` was enabled. Please disable the `log_scale` and try again."
            )

    # Fill the sections under the KDE curve
    if sections is not None and len(sections) > 0:
        for section in sections:
            mask = (kde_values[0] >= section["start"]) & (kde_values[0] <= section["end"])
            plt.fill_between(kde_values[0][mask], kde_values[1][mask], alpha=0.5, label=section["label"])

    # Add a legend
    plt.legend()

    # Show the plot
    plt.xlabel(label_name)
    plt.ylabel("Density")
    return fig.figure


def visualize_distribution_with_outliers(
    values: np.ndarray,
    is_outlier: Optional[List[bool]] = None,
):
    """Visualize the distribution of the data and highlight the potential outliers."""

    if is_outlier is None:
        # Import here to prevent ciruclar imports
        from auroris.curation.functional import detect_outliers

        is_outlier = detect_outliers(values)

    # sort both value and outlier indicator
    sorted_ind = np.argsort(values)
    values = values[sorted_ind]
    is_outlier = is_outlier[sorted_ind]

    with create_figure(n_plots=2) as (fig, axes):
        sns.scatterplot(
            x=np.arange(len(values)),
            y=values,
            hue=is_outlier,
            palette={1.0: "red", 0.0: "navy", 0.5: "grey"},
            ax=axes[0],
        )
        stats.probplot(values, dist="norm", plot=axes[1])

    return fig
