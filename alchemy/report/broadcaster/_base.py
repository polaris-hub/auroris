import abc

from PIL.Image import Image as ImageType

from alchemy.report import CurationReport, Section


class ReportBroadcaster(abc.ABC):
    """
    Creates a specific view of the report.
    Implements the Template Method Design Pattern.
    """

    def __init__(self, report: CurationReport):
        self._report = report

    def broadcast(self):
        self.on_report_start(self._report)
        for section in self._report.sections:
            self.on_section_start(section)

            if len(section.logs) > 0:
                self.on_logs_start()
                for log in section.logs:
                    self.render_log(log)
                self.on_logs_end()

            if len(section.images) > 0:
                self.on_images_start()
                for image in section.images:
                    self.render_image(image)
                self.on_images_end()

            self.on_section_end(section)
        self.on_report_end(self._report)

    @abc.abstractmethod
    def render_log(self, message: str):
        raise NotImplementedError

    @abc.abstractmethod
    def render_image(self, image: ImageType):
        raise NotImplementedError

    def on_section_start(self, section: Section):
        pass

    def on_section_end(self, section: Section):
        pass

    def on_logs_start(self):
        pass

    def on_logs_end(self):
        pass

    def on_images_start(self):
        pass

    def on_images_end(self):
        pass

    def on_report_start(self, report: CurationReport):
        pass

    def on_report_end(self, section: CurationReport):
        pass
