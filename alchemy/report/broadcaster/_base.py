import abc

from alchemy.report import CurationReport, Section


class ReportBroadcaster(abc.ABC):
    """
    Creates a specific view of the report.
    """

    def broadcast(self, report: CurationReport):
        for section in report.sections:
            for log in section.logs:
                self.broadcast_log(log)
            self.section_separator(section)

    @abc.abstractmethod
    def broadcast_log(self, message: str):
        raise NotImplementedError

    @abc.abstractmethod
    def section_separator(self, section: Section):
        raise NotImplementedError
