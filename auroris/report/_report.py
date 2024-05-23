from contextlib import contextmanager
from datetime import datetime
from typing import ByteString, List, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL.Image import Image as ImageType
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from auroris import __version__
from auroris.utils import bytes2img, fig2img


class AnnotatedImage(BaseModel):
    """
    Image data, potentially with a title and / or description.
    """

    image: ImageType
    title: Optional[str] = ""
    description: Optional[str] = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Section(BaseModel):
    """
    A section in a report.
    """

    title: str
    logs: List[str] = Field(default_factory=list)
    images: List[AnnotatedImage] = Field(default_factory=list)


class CurationReport(BaseModel):
    """
    A report that summarizes the changes of the curation process.
    """

    title: str = "Curation Report"
    sections: List[Section] = Field(default_factory=list)
    auroris_version: str = Field(default=__version__)
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
        self._check_active_section()
        self._active_section.logs.append(message)

    def log_new_column(self, name: str):
        """Log that a new column has been added to the dataset"""
        self.log(f"New column added: {name}")

    def log_image(
        self,
        image_or_figure: Union[ImageType, Figure, ByteString],
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Logs an image. Also accepts Matplotlib figures, which will be converted to images."""
        self._check_active_section()
        if isinstance(image_or_figure, Figure):
            image = fig2img(image_or_figure)
            plt.close(image_or_figure)
        elif isinstance(image_or_figure, ByteString):
            image = bytes2img(image_or_figure)
        else:
            image = image_or_figure

        image = AnnotatedImage(image=image, title=title, description=description)
        self._active_section.images.append(image)

    def _check_active_section(self):
        if self._active_section is None:
            raise RuntimeError("No active section. Use `with report.section(name):`")
