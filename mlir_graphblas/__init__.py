import donfig
from ._version import get_versions
from .cli import MlirOptCli, MlirOptError
from .engine import MlirJitEngine
from . import tools

__version__ = get_versions()["version"]
del get_versions


config = donfig.Config("mlir-graphblas")
