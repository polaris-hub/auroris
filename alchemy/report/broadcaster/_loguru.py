from loguru import logger

from alchemy.report import Section

from ._base import ReportBroadcaster


class LoguruBroadcaster(ReportBroadcaster):
    def broadcast_log(self, message: str):
        logger.info(message)

    def section_separator(self, section: Section):
        logger.debug("-" * 50)
