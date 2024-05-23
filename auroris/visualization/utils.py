from contextlib import contextmanager
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


@contextmanager
def create_figure(
    n_plots: int,
    n_cols: Optional[int] = None,
    fig_base_size: float = 8,
    w_h_ratio: float = 0.5,
    dpi: int = 150,
    seaborn_theme: Optional[str] = "whitegrid",
):
    """Creates a figure with the desired size and layout"""

    if seaborn_theme is not None:
        sns.set_theme(style=seaborn_theme)

    if n_cols is None or n_cols > n_plots:
        n_cols = n_plots

    n_rows = n_plots // n_cols
    if n_plots % n_cols > 0:
        n_rows += 1

    fig_w = fig_base_size * n_cols
    fig_h = fig_base_size * w_h_ratio * n_rows

    # Create the figure
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        dpi=dpi,
    )
    axes = np.atleast_1d(axes)
    axes = axes.flatten()
    yield fig, axes

    # Remove unused axes
    _ = [fig.delaxes(a) for a in axes[n_plots:]]
