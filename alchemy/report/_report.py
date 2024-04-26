from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL.Image import Image as ImageType
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from alchemy import __version__
from alchemy.utils import fig2img


class Section(BaseModel):
    """
    A section in a report.
    """

    title: str
    logs: List[str] = Field(default_factory=list)
    images: List[ImageType] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CurationReport(BaseModel):
    """
    A report that summarizes the changes of the curation process.
    """

    sections: List[Section] = Field(default_factory=list)
    alchemy_version: str = Field(default=__version__)
    time_stamp: datetime = Field(default_factory=datetime.now)

    _active_section: Optional[Section] = PrivateAttr(None)

    def start_section(self, name: str):
        self.sections.append(Section(title=name))
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
        """Log a message to the report"""
        self._active_section.logs.append(message)

    def log_new_column(self, name: str):
        """Log that a new column has been added to the dataset"""
        self.log(f"New column added: {name}")

    def log_image(self, image_or_figure: Union[ImageType, Figure]):
        """Logs an image. Also accepts Matplotlib figures, which will be converted to images."""

        if isinstance(image_or_figure, Figure):
            image = fig2img(image_or_figure)
            plt.close(image_or_figure)

        else:
            image = image_or_figure

        self._active_section.images.append(image)
