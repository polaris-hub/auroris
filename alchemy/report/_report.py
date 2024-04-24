from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from alchemy import __version__


class Section(BaseModel):
    """
    A section in a report.
    """

    name: str
    logs: List[str] = Field(default_factory=list)


class CurationReport(BaseModel):
    """
    A report that summarizes the changes of the curation process.
    """

    sections: List[Section] = Field(default_factory=list)
    alchemy_version: str = Field(default=__version__)
    time_stamp: datetime = Field(default_factory=datetime.now)

    _active_section: Optional[Section] = PrivateAttr(None)

    def start_section(self, name: str):
        self.sections.append(Section(name=name))
        self._active_section = self.sections[-1]

    def end_section(self):
        self._active_section = None

    @contextmanager
    def section(self, name: str):
        self.start_section(name)
        try:
            yield self
        finally:
            self.end_section()

    def log(self, message: str):
        self._active_section.logs.append(message)
