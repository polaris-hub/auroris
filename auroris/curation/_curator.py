import json
from typing import List, Tuple, Union, Optional, Annotated

from os import PathLike
import fsspec
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_serializer, field_validator

from auroris.curation.actions import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel


class Curator(BaseModel):
    """
    A curator is a collection of actions that are applied to a dataset.
    Can be serialized.

    """

    # To know which Action object to create, we need a discriminated union.
    # This is the recommended way to add all subclasses in the type.
    # See e.g. https://github.com/pydantic/pydantic/issues/2200
    # and https://github.com/pydantic/pydantic/issues/2036
    src_dataset_path: Optional[Union[str, PathLike]] = Field(
        default=None,
        description="Data path. The data must be loadable by `pd.read_csv` with default parameters.",
    )

    steps: List[
        Annotated[
            Union[tuple(BaseAction.__subclasses__())],
            Field(
                ...,
                discriminator="name",
                description="List of curation actions. Check all the available action <auroris.curation.actions.__all__>.",
            ),
        ]
    ]
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

    @field_validator("src_dataset_path", mode="before")
    def _validate_src_dataset_path(cls, value: Union[str, PathLike]):
        try:
            pd.read_csv(value, nrows=5)
            return value
        except Exception:
            raise ValueError(
                f"Dataset can't be loaded by `pandas.read_csv('{value}')`."
                f"Consider passing the DataFrame directly to `Curator.curate(dataset=...)`."
            )

    def _load_data(self):
        return pd.read_csv(self.src_dataset_path)

    def transform(self, dataset: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, CurationReport]:
        if dataset is None:
            dataset = self._load_data()

        report = CurationReport()
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
        serialization = self.model_dump()
        # remove src_dataset_path if unavailable
        if self.src_dataset_path is None:
            serialization.pop("src_dataset_path")
        with fsspec.open(path, "w") as f:
            json.dump(serialization, f)
        return path
