import json
from typing import Annotated, List, Optional, Tuple, Union

import datamol as dm
import fsspec
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_serializer, field_validator

from auroris.curation.actions import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.utils import is_parquet_file


class Curator(BaseModel):
    """
    A curator is a serializable collection of actions that are applied to a dataset.

    Attributes:
        steps (List[BaseAction]): Ordered list of curation actions to apply to the dataset.
        src_dataset_path: An optional path to load the source dataset from. Can be used to specify a reproducible workflow.
        verbosity: Verbosity level for logging.
        parallelized_kwargs: Keyword arguments to affect parallelization in the steps.
    """

    # To know which Action object to create, we need a discriminated union.
    # This is the recommended way to add all subclasses in the type.
    # See e.g. https://github.com/pydantic/pydantic/issues/2200
    # and https://github.com/pydantic/pydantic/issues/2036
    steps: List[
        Annotated[
            Union[tuple(BaseAction.__subclasses__())],  # type: ignore
            Field(..., discriminator="name"),
        ]
    ]

    src_dataset_path: Optional[str] = None
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    parallelized_kwargs: dict = Field(default_factory=dict)

    @field_validator("verbosity", mode="before")
    def _validate_verbosity(cls, v):
        if not isinstance(v, VerbosityLevel):
            return VerbosityLevel[v]
        return v

    @field_serializer("verbosity")
    def _serialize_verbosity(self, value: VerbosityLevel):
        return value.name

    @field_validator("src_dataset_path")
    def _validate_src_dataset_path(cls, value: Optional[str]):
        # If not set, no need to validate
        if value is None:
            return value

        # Efficient check to see if it's a valid path to a supported file
        if not is_parquet_file(value):
            try:
                pd.read_csv(value, nrows=5)
            except Exception:
                raise ValueError(
                    f"Dataset can't be loaded by `pandas.read_csv('{value}')` nor `pandas.read_parquet('{value}')`."
                    f"Consider passing the DataFrame directly to `Curator.curate(dataset=...)`."
                )

        # If it's set, but local, warn the user that this hinders reproducibility.
        if dm.utils.fs.is_local_path(value):
            logger.warning(
                "Using a local path for `src_dataset_path` hinders reproducibility. "
                "Consider uploading the file to a public cloud storage service."
            )
        return value

    def transform(self, dataset: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, CurationReport]:
        """Runs the curation process.

        Args:
            dataset: The dataset to be curated. If `src_dataset_path` is set, this parameter is ignored.

        Returns:
            A tuple of the curated dataset and a report summarizing the changes made.
        """

        if self.src_dataset_path is not None:
            if dataset is not None:
                logger.warning(
                    "Both `self.scr_dataset_path` and the `dataset` parameter are specified. "
                    "Ignoring the `dataset` parameter."
                )

            dataset = self.load_dataset(self.src_dataset_path)

        if dataset is None:
            raise ValueError("Running the curator requires a source dataset.")

        # The report summarizes the changes made to the dataset
        report = CurationReport()

        # Changes are not made in place
        dataset = dataset.copy(deep=True)

        action: BaseAction
        for action in self.steps:
            logger.info(f"Performing step: {action.name}")

            with report.section(action.name):
                dataset = action.transform(
                    dataset,
                    report=report,
                    verbosity=self.verbosity,
                    parallelized_kwargs=self.parallelized_kwargs,
                )

        return dataset, report

    @staticmethod
    def load_dataset(path: str):
        """
        Loads a dataset, to be curated, from a path.

        Info: File-format support
            This currently only supports CSV and Parquet files and uses the default
            parameters for `pd.read_csv` and `pd.read_parquet`. If you need more flexibility,
            consider loading the data yourself and passing it directly to `Curator.transform(dataset=...)`.
        """
        if not is_parquet_file(path):
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def __call__(self, dataset):
        return self.transform(dataset)

    @classmethod
    def from_json(cls, path: str):
        """Loads a curation workflow from a JSON file.

        Args:
            path:The path to load from
        """
        with fsspec.open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str):
        """Saves the curation workflow to a JSON file.

        Args:
            path: The destination to save to.
        """
        with fsspec.open(path, "w") as f:
            json.dump(self.model_dump(), f)
