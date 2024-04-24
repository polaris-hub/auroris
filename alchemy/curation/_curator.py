import json
from typing import List

import fsspec
from pydantic import BaseModel

from alchemy.curation import CuratorConstants
from alchemy.curation.actions import Action


class Curator(BaseModel):
    """
    A curator is a collection of actions that are applied to a dataset.
    Can be serialized.
    """

    actions: List[Action]
    constants: CuratorConstants

    def run(self, dataset):
        for action in self.actions:
            dataset = action.run(dataset)
        return dataset

    def __call__(self, dataset):
        return self.run(dataset)

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
