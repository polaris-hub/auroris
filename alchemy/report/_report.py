from typing import Dict, List

from pydantic import BaseModel, PrivateAttr


class CurationReport(BaseModel):
    """
    A report that summarizes the changes of the curation process.
    """

    logs: Dict[str, List[str]]

    _active_section: str = PrivateAttr(None)

    def log(self, message: str):
        self.logs[self._active_section].append(message)
