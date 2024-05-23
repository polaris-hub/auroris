from enum import IntEnum


class VerbosityLevel(IntEnum):
    """The different verbosity levels"""

    SILENT = 0
    NORMAL = 1
    VERBOSE = 2
    DEAFENING = 3
