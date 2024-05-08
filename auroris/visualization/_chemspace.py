from typing import Any, List, Optional, Union

import numpy as np
import seaborn as sns

from auroris.visualization.utils import create_figure

try:
    import umap  # type: ignore
except ImportError:
    umap = None


def visualize_chemspace(
    X: Union[List[np.ndarray], np.ndarray],
    y: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    n_cols: int = 3,
    fig_base_size: float = 8,
    w_h_ratio: float = 0.5,
    dpi: int = 150,
    seaborn_theme: Optional[str] = "whitegrid",
    **umap_kwargs: Any,
):
    """Plot the chemical space. Also, color based on the target values.

    Args:
        X: A list of arrays with the features.
        y: A list of arrays with the target values.
        labels: Optional list of labels for each set of features.
        n_cols: Number of columns in the subplots.
        fig_base_size: Base size of the plots.
        w_h_ratio: Width/height ratio.
        dpi: DPI value of the figure.
        seaborn_theme: Seaborn theme.
        **umap_kwargs: Keyword arguments for the UMAP algorithm.
    """

    if umap is None:
        raise ImportError("Please run `pip install umap-learn` to use UMAP visualizations for the chemspace.")

    if isinstance(X, np.ndarray):
        X = [X]
    if isinstance(y, np.ndarray):
        y = [y]
    if y is None:
        y = [None for _ in range(len(X))]
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    if labels is None:
        labels = ["" for i in range(len(X))]

    with create_figure(
        n_plots=len(X),
        n_cols=n_cols,
        fig_base_size=fig_base_size,
        w_h_ratio=w_h_ratio,
        dpi=dpi,
        seaborn_theme=seaborn_theme,
    ) as (fig, axes):
        for idx, (X_i, y_i, label) in enumerate(zip(X, y, labels)):
            embedding = umap.UMAP(**umap_kwargs).fit_transform(X_i)
            umap_0, umap_1 = embedding[:, 0], embedding[:, 1]

            ax = sns.scatterplot(
                x=umap_0,
                y=umap_1,
                hue=y_i,
                ax=axes[idx],
            )
            ax.set_title(label)

    return fig
