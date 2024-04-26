import datamol as dm
import fsspec
from PIL.Image import Image as ImageType

from alchemy.report import CurationReport, Section

from ._base import ReportBroadcaster


class HTMLBroadcaster(ReportBroadcaster):
    """Render a simple HTML page"""

    def __init__(self, report: CurationReport, destination: str):
        super().__init__(report)

        # Create destination dir.
        self._destination = destination
        dm.fs.mkdir(self._destination, exist_ok=True)

        # Create the directory for images
        self._image_dir = dm.fs.join(self._destination, "images")
        dm.fs.mkdir(self._image_dir, exist_ok=True)

        # Open the file
        path = dm.fs.join(self._destination, "report.html")
        self._file_descriptor = fsspec.open(path, "w")
        self._file = self._file_descriptor.__enter__()

        # Count the images
        self._image_count = 0

    def __del__(self):
        # Close the file
        self._file_descriptor.__exit__()

    def on_logs_start(self):
        # Start an unordered list
        self._file.write("<h3>Logs</h3>")
        self._file.write("<ul>")

    def on_logs_end(self):
        # Start an unordered list
        self._file.write("</ul>")

    def on_images_start(self):
        self._file.write("<h3>Images</h3>")

    def render_log(self, message: str):
        # Write the log to the HTML
        self._file.write(f"<li>{message}</li>")

    def render_image(self, image: ImageType):
        # Save the image to the disk
        path = dm.fs.join(self._image_dir, f"image_{self._image_count}.png")
        image.save(path)

        # Write the image to the HTML
        self._file.write(f'<img src="{path}" />')

        # Increase counter
        self._image_count += 1

    def on_report_start(self, report: CurationReport):
        self._file.write("<html>")
        self._file.write("<h1>Curation Report</h1>")
        self._file.write(f"<p>Time: {report.time_stamp.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        self._file.write(f"<p>Version: {report.alchemy_version}</p>")

    def on_section_start(self, section: Section):
        self._file.write(f"<h2>{section.title}</h2>")

    def on_report_end(self, report: CurationReport):
        self._file.write("</html>")
