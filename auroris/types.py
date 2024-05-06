from enum import IntEnum
from PIL.Image import Image


class VerbosityLevel(IntEnum):
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2
    DEAFENING = 3
