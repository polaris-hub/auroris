from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("auroris")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
