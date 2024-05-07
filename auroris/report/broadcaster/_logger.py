import logging
import sys

from typing import Union
from PIL.Image import Image as ImageType

from auroris.report import CurationReport, Section

from ._base import ReportBroadcaster
from auroris.utils import get_image_dimension


class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;1m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        formatter = logging.Formatter(self.FORMATS[record.levelno])
        return formatter.format(record)


class LoggerBroadcaster(ReportBroadcaster):
    """Simple broadcaster for debugging that will simply write to the terminal"""

    def __init__(self, report: CurationReport):
        super().__init__(report)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        self.logger.addHandler(handler)

    def render_log(self, message: str):
        self.logger.debug(f"[LOG]: {message}")

    def render_image(self, image: Union[ImageType, bytes]):
        if isinstance(image, bytes):
            width, height = get_image_dimension(image)
        else:
            width, height = image.width, image.height
        self.logger.debug(f"[IMG]: Dimensions {width} x {height}")

    def on_section_start(self, section: Section):
        self.logger.info(f"===== {section.title} =====")

    def on_report_start(self, report: CurationReport):
        self.logger.critical("===== Curation Report =====")
        self.logger.debug(f"Time: {report.time_stamp.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.debug(f"Version: {report.auroris_version}")

    def on_image_start(self, message: str = None):
        pass

    def on_image_end(self, message: str = None):
        pass
