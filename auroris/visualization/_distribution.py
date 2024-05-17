from typing import List, Optional, Sequence

import numpy as np
import seaborn as sns
from scipy import stats

from auroris.visualization.utils import create_figure


def visualize_continuous_distribution(
    data: np.ndarray, log_scale: bool = False, bins: Optional[Sequence[float]] = None
):
    """
    KDE plot the distribution of the column in `data` with colored sections under the KDE curve.

    Args:
        data: A 1D numpy array with the values to plot the distribution for.
        log_scale: Whether to plot the x-axis in log scale.
        bins: The bin boundaries to color the area under the KDE curve.
    """
    # Create a KDE plot without filling
    with create_figure(n_plots=1) as (fig, axs):
        ax = sns.kdeplot(data, ax=axs[0], log_scale=log_scale, color="black", linewidth=1.5)

    if bins is None:
        return fig

    # Get the xy coordinates of the plotted KDE line
    coords = ax.get_lines()[0].get_data()
    xs = coords[0]
    ys = coords[1]

    # Setup the bins
    bins = np.sort(bins)
    bins = np.append(bins, np.inf)
    lower = -np.inf

    ylim = ax.get_ylim()

    # Color the area under the KDE curve in accordance with the bins
    # Also added a vertical dashed line for each bin boundary
    for threshold in bins:
        if log_scale and lower != -np.inf:
            lower = np.log(lower)

        if log_scale and threshold != np.inf:
            threshold = np.log(threshold)

        mask = (xs > lower) & (xs <= threshold)

        # Update xs to make sure they cover the range even if the
        # coordinates don't fully cover it
        masked_xs = xs[mask]

        if len(masked_xs) == 0:
            continue

        masked_xs[0] = max(lower, np.min(xs))
        masked_xs[-1] = threshold

        pct_mask = (data > lower) & (data <= threshold)
        pct = np.sum(pct_mask) / len(data)

        def _format(val):
            if val == -np.inf:
                return "-∞"
            elif val == np.inf:
                return "∞"
            else:
                return f"{val:.2f}"

        label = f"{_format(lower)}, {_format(threshold)}"
        label = f"({label})" if threshold == np.inf else f"({label}]"
        label += f" - {pct:.2%}"

        ax.fill_between(masked_xs, ys[mask], alpha=0.5, label=label)
        ax.plot([threshold, threshold], [ylim[0], ys[mask][-1]], "k--")
        lower = threshold

    ax.legend()
    return fig


def visualize_distribution_with_outliers(
    values: np.ndarray, is_outlier: Optional[List[bool]] = None, title: str = "Probability Plot"
):
    """
    Visualize the distribution of the data and highlight the potential outliers.

    Args:
        values: Values for visulization.
        is_outlier: List of outlier flag.
        title: Title of plot

    """

    if is_outlier is None:
        # Import here to prevent ciruclar imports
        from auroris.curation.functional import detect_outliers

        is_outlier = detect_outliers(values)

    # sort both value and outlier indicator
    sorted_ind = np.argsort(values)
    values = values[sorted_ind]
    is_outlier = is_outlier[sorted_ind]

    with create_figure(n_plots=1) as (fig, axes):
        res = stats.probplot(values, dist="norm", fit=True, plot=axes[0])
        x = res[0][0]
        y = res[0][1]

        # Specify the indices of data points to highlight
        highlight_indices = np.argwhere(is_outlier.__eq__(True)).flatten()
        highlight_color = "red"

        # Overlay specific points with different colors
        for idx in highlight_indices:
            axes[0].plot(
                x[idx], y[idx], marker="o", markersize=8, color=highlight_color
            )  # Red circles for highlighted points

        axes[0].set_xlabel("Theoretical quantiles")
        axes[0].set_ylabel("Ordered Values")
        axes[0].set_title(title)

    return fig
