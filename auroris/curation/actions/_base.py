import abc
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd
from pydantic import BaseModel, model_validator

from auroris.types import VerbosityLevel

if TYPE_CHECKING:
    from auroris.report import CurationReport


class BaseAction(BaseModel, abc.ABC):
    """
    An action in the curation process.

    Info: The importance of reproducibility
        One of the main goals in designing `auroris` is to make it easy to reproduce the curation process.
        Reproducibility is key to scientific research. This is why a BaseAction needs to be serializable and
        uniquely identified by a `name`.

    Attributes:
        name: The name that uniquely identifies the action. This is used to serialize and deserialize the action.
        prefix: This prefix is used when an action adds columns to a dataset.
            If not set, it defaults to the name in uppercase.
    """

    name: str
    prefix: str = None

    @model_validator(mode="after")
    @classmethod
    def _validate_model(cls, m: "BaseAction"):
        if m.prefix is None:
            m.prefix = m.name.upper() + "_"
        return m

    def get_column_name(self, column: str):
        return f"{self.prefix}{column}"

    @abc.abstractmethod
    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional["CurationReport"] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ):
        raise NotImplementedError

    def __call__(self, dataset: pd.DataFrame):
        return self.transform(dataset)
