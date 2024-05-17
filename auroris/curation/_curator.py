import json
from typing import List, Tuple, Union, Optional

from os import PathLike
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
    data_path: Optional[Union[str, PathLike]] = Field(
        default=None,
        description="Data path. The data must be loadable by `pd.read_csv` with default parameters.",
    )

    steps: List[Union[tuple(ACTION_REGISTRY)]] = Field(
        ...,
        discriminator="name",
        description="List of curation actions. Check all the available action <auroris.curation.actions.__all__>.",
    )
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

    @field_validator("data_path", mode="before")
    def _validate_data_path(cls, value: Union[str, PathLike]):
        try:
            pd.read_csv(value, nrows=5)
            return value
        except Exception:
            raise ValueError(
                f"Dataset can't be loaded by `pandas.read_csv('{value}')`."
                f"Consider passing the DataFrame directly to `Curator.curate(dataset=...)`."
            )

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def transform(self, dataset: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, CurationReport]:
        if dataset is None:
            dataset = self._load_data()

        report = CurationReport()
        dataset = dataset.copy()

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
    def _get_action(cls, name: str):
        for action in ACTION_REGISTRY:
            if action.__name__ == name:
                return action
        return None

    @classmethod
    def from_json(cls, path: str):
        """Loads a curation workflow from a JSON file.

        Args:
            path:The path to load from
        """
        with fsspec.open(path, "r") as f:
            data = json.load(f)

        steps = [cls._get_action(name)(**args) for step in data["steps"] for name, args in step.items()]
        data["steps"] = steps
        return cls.model_validate(data)

    def to_json(self, path: str):
        """Saves the curation workflow to a JSON file.

        Args:
            path: The destination to save to
        """
        serialization = self.model_dump(exclude="steps")
        # remove data_path
        if self.data_path is None:
            serialization.pop("data_path")
        # save steps in defined order
        serialization["steps"] = [{step.name: step.model_dump()} for step in self.steps]
        with fsspec.open(path, "w") as f:
            json.dump(serialization, f)
        return path
