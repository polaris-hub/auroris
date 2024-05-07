import json
from typing import List, Tuple, Union

import fsspec
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_serializer, field_validator

from auroris.curation.actions._base import ACTION_REGISTRY, BaseAction
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
    steps: List[Union[tuple(ACTION_REGISTRY)]] = Field(..., discriminator="name")  # type: ignore

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

    def transform(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, CurationReport]:
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
        return cls.model_validate(data)

    def to_json(self, path: str):
        """Saves the curation workflow to a JSON file.

        Args:
            path: The destination to save to
        """
        with fsspec.open(path, "w") as f:
            json.dump(self.model_dump(), f)
        return path
