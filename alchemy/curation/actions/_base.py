import abc

import pandas as pd
from pydantic import BaseModel


class BaseAction(BaseModel, abc.ABC):
    """
    An action in the curation process.
    """

    @abc.abstractmethod
    def run(self, dataset: pd.DataFrame):
        raise NotImplementedError

    def __call__(self, dataset: pd.DataFrame):
        return self.run(dataset)
