import donfig
from . import cli
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


config = donfig.Config("mlir-graphblas")
