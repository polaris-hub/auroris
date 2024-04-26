from loguru import logger
from PIL.Image import Image as ImageType

from alchemy.report import CurationReport, Section

from ._base import ReportBroadcaster


class LoguruBroadcaster(ReportBroadcaster):
    """Simple broadcaster for debugging that will simply write to the terminal"""

    def __init__(self, report: CurationReport):
        super().__init__(report)

    def render_log(self, message: str):
        logger.info(f"[LOG]: {message}")

    def render_image(self, image: ImageType):
        logger.info(f"[IMG]: {image.width} x {image.height}")

    def on_section_start(self, section: Section):
        logger.info(f"===== {section.title} =====")

    def on_report_start(self, report: CurationReport):
        logger.info(f"Time: {report.time_stamp.strftime("%Y-%m-%d %H:%M:%S")}")
        logger.info(f"Version: {report.alchemy_version}")
