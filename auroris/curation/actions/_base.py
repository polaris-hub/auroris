import abc
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd
from pydantic import BaseModel, model_validator

from auroris.types import VerbosityLevel

if TYPE_CHECKING:
    from auroris.report import CurationReport


ACTION_REGISTRY = []


class BaseAction(BaseModel, abc.ABC):
    """
    An action in the curation process.

    Args:
        prefix: If the action adds columns, use this prefix.
        completed: If the action has completed.
        dep_action: Name of dependent action.
    """

    prefix: str = None

    @property
    def name(self) -> str:
        """The name of the action. Needs to be unique."""
        return self.__class__.__name__

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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ACTION_REGISTRY.append(cls)
