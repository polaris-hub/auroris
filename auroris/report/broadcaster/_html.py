import base64
import os
import pathlib
from copy import deepcopy
from importlib import resources

import datamol as dm
import fsspec

from auroris.report import CurationReport
from auroris.utils import img2bytes

from ._base import ReportBroadcaster

try:
    import jinja2
except ImportError:
    jinja2 = None


class HTMLBroadcaster(ReportBroadcaster):
    """Render a simple HTML page"""

    def __init__(
        self,
        report: CurationReport,
        destination: str,
        embed_images: bool = False,
    ):
        super().__init__(report)

        if jinja2 is None:
            raise ImportError(
                f"Jinja2 is required to use {self.__class__.__name__}. Install it with `pip install Jinja2`."
            )

        self._destination = destination
        self._image_dir = dm.fs.join(self._destination, "images")
        self._embed_images = embed_images

    def broadcast(self):
        report = deepcopy(self._report)

        # Create destination dir.
        dm.fs.mkdir(self._destination, exist_ok=True)

        # Create the directory for images
        if not self._embed_images:
            dm.fs.mkdir(self._image_dir, exist_ok=True)

        pathlib.Path(__file__).parent.resolve() / "templates"

        # Save all images
        image_counter = 0
        for section in report.sections:
            for image in section.images:
                if self._embed_images:
                    # Encode directly into the HTML
                    image_data = img2bytes(image.image)
                    image_data = base64.b64encode(image_data).decode("utf-8")
                    src = f"data:image/png;base64,{image_data}"
                else:
                    # Save as separate file
                    path = dm.fs.join(self._image_dir, f"{image_counter}.png")
                    image.image.save(path)
                    src = os.path.relpath(path, self._destination)

                image.image = src
                image_counter += 1

        # Get HTML template file
        path = resources.files("auroris.report.broadcaster.templates")
        path = path.joinpath("report.html.jinja")

        # Render the HTML
        with path.open() as template_file:
            template = jinja2.Template(template_file.read())
        html = template.render(report=report)

        # Write the HTML
        path = dm.fs.join(self._destination, "index.html")
        with fsspec.open(path, "w") as fd:
            fd.write(html)
