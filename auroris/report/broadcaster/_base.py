import abc

from auroris.report import CurationReport


class ReportBroadcaster(abc.ABC):
    """
    Creates a specific view of the report.
    Implements the Template Method Design Pattern.
    """

    def __init__(self, report: CurationReport):
        self._report = report

    @abc.abstractmethod
    def broadcast(self):
        raise NotImplementedError
