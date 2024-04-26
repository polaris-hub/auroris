from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats

from alchemy.visualization.utils import create_figure


def detailed_distributions_plots(
    df: pd.DataFrame,
    thresholds: Dict[str, Tuple[int, Callable]] = None,
    label_names: List[str] = None,
    log_scale_mapping: Dict[str, bool] = None,
    positive_color: str = "#3db371",
    negative_color: str = "#a9a9a9",
    n_cols: int = 3,
    fig_base_size: float = 8,
    w_h_ratio: float = 0.5,
    legend_fontsize: int = 18,
    ticks_fontsize: int = 18,
    title_fontsize: int = 18,
    gridsize: int = 1000,
    dpi: int = 150,
    seaborn_theme: Optional[str] = "whitegrid",
):
    """Plot the detailed distribution of the columns in `df`. Also, color the part of the
    "positive" distribution using `thresholds`.

    Args:
        df: A dataframe with binarized readouts only. NaN are allowed.
        thresholds: A dict mapping of the `df` column. Value is a tuple where the first
            element is the threshold value and the second element is a callable deciding wether
            a datapoint meets the criterai or not (something like `np.less` or np.greater`).
        label_names: Name of the labels (same order as the columns in `df`). If not set
            the name of the columns are used.
        log_scale_mapping: A dict mapping of the `df` column. If True,
            the plot for this readout will be log scaled.
        positive_color: Color for `True` or `1`.
        negative_color: Color for `False` or `0`.
        n_cols: Number of columns in the subplots.
        fig_base_size: Base size of the plots.
        w_h_ratio: Width/height ratio.
        legend_fontsize: Font size of the legend.
        ticks_fontsize: Font size of the x ticks and x label.
        title_fontsize: Font size of the title.
        gridsize: Gridsize for the kernel density estimate (KDE).
        dpi: DPI value of the figure.
        seaborn_theme: Seaborn theme.
    """

    # NOTE: the `thresholds` API is not super nice, consider an alternative.
    # NOTE: we could eventually add support for multiclass here if we need it.
    if thresholds is None:
        thresholds = {}

    if log_scale_mapping is None:
        log_scale_mapping = {}

    if label_names is None:
        label_names = df.columns.tolist()

    # Check all columns are numeric
    numerics = df.apply(lambda x: x.dtype.kind in "biufc")
    if not numerics.all():
        raise ValueError(f"Not all columns are numeric: {numerics[~numerics].to_dict()}")

    n_plots = len(df.columns)

    # Create the figure
    with create_figure(
        n_plots=n_plots,
        n_cols=n_cols,
        dpi=dpi,
        fig_base_size=fig_base_size,
        w_h_ratio=w_h_ratio,
        seaborn_theme=seaborn_theme,
    ) as (fig, axes):
        for ax, readout, label_name in zip(axes, df.columns, label_names):
            values = df[readout].dropna()

            # Get threshold value and function
            threshold_value, threshold_fn = None, None
            threshold = thresholds.get(readout, None)
            if threshold is not None:
                threshold_value, threshold_fn = threshold

            # Whether to log scale
            log_scale = log_scale_mapping.get(readout, False)

            # Draw distribution and kde plot
            kde_kws = {}
            kde_kws["clip"] = values.min(), values.max()
            kde_kws["gridsize"] = gridsize
            kplot = sns.histplot(
                values,
                kde=True,
                ax=ax,
                color=negative_color,
                kde_kws=kde_kws,
                log_scale=log_scale,
            )

            # Label
            ax.set_title(label_name, fontsize=title_fontsize)
            ax.set_xlabel(None)
            ax.set_ylabel("Count", fontsize=ticks_fontsize)

            ax.xaxis.set_tick_params(labelsize=ticks_fontsize)
            ax.yaxis.set_tick_params(labelsize=ticks_fontsize)

            if threshold_value is not None and threshold_fn is not None:
                # Fill between on active values
                x, y = kplot.get_lines()[0].get_data()
                ax.fill_between(
                    x,
                    y,
                    where=threshold_fn(x, threshold_value),
                    facecolor=positive_color,
                    alpha=0.8,
                )

                # Active ratio text box
                positive_ratio = threshold_fn(values, threshold_value).sum() / len(values) * 100
                ax.text(
                    0.85,
                    0.95,
                    f"{positive_ratio:.1f} %",
                    transform=ax.transAxes,
                    fontsize=legend_fontsize,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
            else:
                logger.warning(f"Threshold not available for readout '{readout}'")

    return fig


def visualize_distribution_with_outliers(
    values: np.ndarray,
    is_outlier: Optional[List[bool]] = None,
):
    """Visualize the distribution of the data and highlight the potential outliers."""

    values = np.sort(values)

    if is_outlier is None:
        # Import here to prevent ciruclar imports
        from alchemy.curation.functional import detect_outliers

        is_outlier = detect_outliers(values)

    with create_figure(n_plots=2) as (fig, axes):
        sns.scatterplot(
            x=np.arange(len(values)),
            y=values,
            hue=is_outlier,
            palette={True: "red", False: "navy"},
            ax=axes[0],
        )
        stats.probplot(values, dist="norm", plot=axes[1])

    return fig
