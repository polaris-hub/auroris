import abc

from alchemy.report import CurationReport


class ReportBroadcaster(abc.ABC):
    """
    Creates a specific view of the report.
    """

    @abc.abstractmethod
    def broadcast(self, report: CurationReport):
        raise NotImplementedError
