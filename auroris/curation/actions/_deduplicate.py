from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel


def deduplicate(
    dataset: pd.DataFrame,
    deduplicate_on: Optional[Union[str, List[str]]] = None,
    y_cols: Optional[Union[str, List[str]]] = None,
    keep: Literal["first", "last"] = "first",
    method: Literal["mean", "median"] = "median",
) -> pd.DataFrame:
    """
    Deduplicate a dataframe.

    If `deduplicate_on` specifies a subset of all columns in the dataset and `y_cols` specifies a set
    of non-overlapping columns, data will be grouped by `deduplicate_on` and the `y_cols` will be aggregated
    to a single value per group according to `method`.

    Args:
        dataset: The dataset to deduplicate.
        deduplicate_on: A subset of the columns to deduplicate on (can be default).
        y_cols: The columns to aggregate.
        keep: Whether to keep the first or last copy of the duplicates.
        method: The method to aggregate the data.
    """

    groups = []

    if y_cols is None or deduplicate_on is None:
        return dataset.drop_duplicates(subset=deduplicate_on, keep=keep).reset_index(drop=True)

    if len(set(y_cols).intersection(set(deduplicate_on))) > 0:
        raise ValueError("y_cols and deduplicate_on must be non-overlapping.")

    for _, df in dataset.groupby(by=deduplicate_on):
        data_vals = df[y_cols].agg(method, axis=0, skipna=True)
        if isinstance(y_cols, list):
            data_vals = data_vals.tolist()

        df.loc[:, y_cols] = data_vals
        groups.append(df)

    merged_df = pd.concat(groups).sort_values(by=deduplicate_on)
    merged_df = merged_df.drop_duplicates(subset=deduplicate_on, keep=keep).reset_index(drop=True)
    return merged_df


class Deduplication(BaseAction):
    """
    Automatic detection of outliers.

    See [`auroris.curation.functional.deduplicate`][] for the docs of the
    `deduplicate_on`, `y_cols`, `keep` and `method` attributes
    """

    name: Literal["deduplicate"] = "deduplicate"

    deduplicate_on: Optional[Union[str, List[str]]] = None
    y_cols: Optional[Union[str, List[str]]] = None
    keep: Literal["first", "last"] = "first"
    method: Literal["mean", "median"] = "median"

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        dataset_dedup = deduplicate(
            dataset,
            deduplicate_on=self.deduplicate_on,
            y_cols=self.y_cols,
            keep=self.keep,
            method=self.method,
        )
        if report is not None:
            num_duplicates = len(dataset) - len(dataset_dedup)
            report.log(f"Deduplication merged and removed {num_duplicates} duplicated molecules from dataset")
        return dataset_dedup
