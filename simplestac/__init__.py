from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("simplestac")
except PackageNotFoundError:
    # package is not installed
    pass